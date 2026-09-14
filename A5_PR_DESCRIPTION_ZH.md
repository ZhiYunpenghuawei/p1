# Beam Search V1 A5：完整的持久化 GPU 最终输出路径

## 概要

本 PR 基于最新的 A4 / PR 385，实现 Beam Search V1 A5 的完整输出闭环：Beam 状态、候选选择、累计分数、父子关系和 KV 重排继续保留在 GPU 上，最终候选也直接在 GPU 上排序并回溯生成路径；Worker 随后只进行一次紧凑的 D2H 传输，前端再将结果转换为现有离线和在线 API 的公共输出类型。

该实现的核心目标是缩短 Beam Search V1 的关键路径，同时保持现有请求协议、公开输出语义、异常传播和资源生命周期的兼容性。

## 背景与依赖

该 PR 构建在已经完成的分阶段改造之上：

- A2 提供 Beam Search V1 的批处理协议与执行契约。
- A3 将一次 Beam 请求映射为连续的异步 Prefill / Decode stage。
- A4 / PR 385 将持久化 GPU Beam 状态连接到 V1 异步调度和模型执行路径。
- 本 PR（A5）补齐终态选择、结果传输、公共输出转换和资源释放，使 `beam_search_v1` 可以独立完成端到端执行。

本 PR 的基线提交为 PR 385 的 `632f7d2`。

## 主要改动

### 1. GPU 端最终候选选择与路径回溯

新增 `GPUBeamFinalOutput`，直接消费持久化 `BeamSearchState` 中的完成态数据：

- 根据 `completed_scores` 在 GPU 上选择最终 Top-K Beam。
- 沿 append-only parent lineage 回溯完整生成 token 和父 Beam 索引。
- 同时保留累计 logprob、有效长度、结束原因和 stop token。
- 将最终结果直接写入固定布局的紧凑 wire buffer。

最终阶段不再把完整 Beam 状态复制到 Host 后再进行 Python 排序、校验和路径重建。

### 2. 提前提交最终 D2H

对于常见的固定长度结束路径，最后一个 sample 完成时已经可以确定请求终态。本 PR 在该时刻立即提交：

1. GPU 最终 Top-K；
2. GPU 路径回溯和结果 packing；
3. copy stream 上的异步 D2H。

Host 后续消费 terminal receipt 时通常只需要等待已经在执行的拷贝，而不需要再串行启动一次 GPU 工作和 D2H。无法在 sample 时提前确定的 early-stop 路径仍由 `consume()` 安全兜底。

每个成功请求只传输一份最终紧凑 payload，并记录 `final_d2h_count` 和 `final_d2h_bytes` 供诊断使用。

### 3. 离线与在线公共输出转换

新增统一的 `BeamSearchOutputProcessor`：

- 离线路径输出现有 `BeamSearchOutput` / `BeamSearchSequence`。
- 在线路径输出现有 `RequestOutput` / `CompletionOutput`。
- 处理 begin token、end token、stop token 和 `include_stop_str_in_output` 的边界语义。
- 按请求选项保留累计 logprob 和父 Beam 索引。
- 保留 Worker 返回的结构化失败，不将 Engine 错误静默转换为普通空结果。

离线路径复用 PR 380 引入的 `_LazyMaterializedBeamSequence`。请求返回时只保存生成 suffix 和共享 prompt；完整 token 列表和文本在调用方第一次访问相应字段时才恢复，从而避免在关键路径上为所有 Beam 执行长 prompt 复制和 detokenization。

### 4. 终态结果与会话生命周期绑定

最终输出可能仍在 GPU 或 copy stream 上执行，因此会话不能在调度器看到 native request 结束时立即释放。本 PR 明确了以下生命周期：

- terminal receipt 在消费完成前持有最终输出资源。
- `request_release()` 将释放请求标记为 pending。
- 只有 dispatch、control receipt 和 final-output consumer 全部 drain 后才释放 session、KV slot 和共享 workspace。
- 取消、失败、重复 close 和异步清理采用幂等路径。
- 同一 SchedulerOutput 同时退休旧请求并接纳新请求时，先退休旧 owner，避免两个活动 GR session 竞争共享 execution scratch。

### 5. 配置与提交边界

- 安装 canonical `vllm_gr_config` 后清除 Driver 侧旧的解析快照，防止后续消费者继续读取过期配置。
- 请求提交前确认 A5 output conversion 已经安装并且 tokenizer 可用。
- 保持现有 Beam batch protocol 和 EngineCore terminal result 类型不变。

## 端到端执行流程

```text
公共 beam_search_v1 调用
        │
        ▼
输入规范化与 BeamBatchRequest 构建
        │
        ▼
V1 异步调度：Prefill → Decode stage × N
        │
        ▼
GPUBeamStageRunner
  ├─ 原生 Prefill forward
  ├─ 持久化 Decode forward / BeamAttention
  ├─ Constraint Table Top-K
  ├─ GPU Beam 分数累计与父节点选择
  └─ GPU KV gather/scatter
        │
        ▼
最后一个 sample 提前 enqueue final output
  ├─ 完成候选 Top-K
  ├─ parent lineage 回溯
  ├─ 紧凑结果 packing
  └─ 单次异步 D2H
        │
        ▼
Host 消费 terminal receipt
        │
        ▼
BeamBatchResult → 离线/在线公共输出
        │
        ▼
按需懒恢复 tokens / text，随后释放会话资源
```

## 数据所有权与同步约束

- `BeamSearchState` 拥有请求执行期间的 GPU Beam 状态和完成候选。
- `GPUBeamFinalOutput` 拥有最终 device buffer、pinned host buffer、selection scratch 和同步 event。
- 最后一个 producer stream 记录 packing ready event；copy stream 等待该 event 后执行 D2H。
- consumer 完成后记录 done event；session/KV/workspace 只有在该 event 可安全接续后才能复用。
- 公共输出只持有普通 Host 数据，不持有活动 session 或 GPU workspace。

## 兼容性与当前限制

- 保持现有 `beam_search_v1` 离线和在线入口不变。
- 保持 begin/end/stop token 的公共输出语义。
- 保持累计 logprob 和可选 parent lineage 输出。
- 当前持久化快速路径仍要求 CUDA、V1 async、单 TP/PP/DP、FP16/BF16、固定 Beam 容量、Constraint Table 和所有层使用 Beam Attention。
- 当前 A5 terminal payload 只携带最终累计 logprob，不携带逐 token logprob；因此离线 `BeamSearchSequence.logprobs` 当前为空。依赖逐步 logprob 的调用方需要后续扩展 wire contract。
- 当前本地 A/B 中存在 Decode 数值路径差异：Top-1 和 Top-16 候选集合保持一致，但中尾部累计分数变化会导致 rank 重排。该差异来自最终输出处理之前，后续应继续针对 persistent Decode attention/KV 路径进行逐 stage 对齐。

## 关键回归测试

本 PR 增加或更新少量关键测试，覆盖：

- GPU 最终 compaction 保持 score 排名和 parent lineage。
- stop/length 结束原因及输出选项的转换。
- 离线与在线公共输出转换。
- Worker failure 的原样传播。
- terminal receipt 尚未消费时延迟释放 session。
- 同一调度帧中先释放旧 execution owner，再接纳新 GR session。
- 请求提交前 output processor readiness。

## 本地性能结果

测试配置：OneRec-1.7B、BF16、CUDA Graph、Beam width 128、prompt length 1024、生成 3 步、prefix-cache miss、2 次 warmup、5 次采样。

| 指标 | Legacy | A5 |
|---|---:|---:|
| 请求平均延迟 | 108.315 ms | 58.167 ms |
| 请求延迟加速 | — | 1.862× |
| 请求外完整物化平均耗时 | 4.955 ms | 3.311 ms |
| 请求加完整物化平均耗时 | 113.270 ms | 61.478 ms |

Profiling 文件： [https://github.com/ZhiYunpenghuawei/p1/tree/main/profiling]()

以上结果仅代表当前本地环境，最终数据应以目标机器上的独立复测为准。

## 本地精度对比

在相同配置和输入下，5 次采样中两条路径各自完全稳定：

- Top-1 序列一致率：100%。
- Top-4 / Top-8 / Top-16 候选集合一致率：100%。
- 128 个最终候选中有 125 个相同，候选集合重合率为 97.656%。
- 同 rank token 序列一致率为 11.719%，第一次顺序差异出现在零起始 rank 4；该指标主要反映 score 轻微变化造成的排序放大。
- 对相同 token 序列重新对齐后，累计 score 绝对误差中位数为 0.01116，P95 为 0.12127，最大值为 0.16122。

因此，当前结果不是候选集合整体失效，而是 persistent Decode 数值差异引起的中尾部排序变化。该问题不由 lazy materialization、最终 GPU packing 或 D2H 引入；这些终态改造前后的候选重合率和 score 误差保持不变。

## Review 重点

建议重点检查：

- 最后一个 sample 与 final-output enqueue 之间的 stream/event 顺序。
- early stop、失败和取消路径是否都只交付一次 terminal result。
- terminal consumer 与 session/KV/workspace 释放之间的所有权边界。
- GPU lineage 回溯对短 stop completion、重复 parent 和固定长度 completion 的处理。
- 离线/在线 begin、end 和 stop token 的一致性。

