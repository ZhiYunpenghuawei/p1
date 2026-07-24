# 基于 xAttention 的动态 Beam Width 适配分析

## 1. 范围说明

本文所说的动态 Beam Width（动态 BW）是指：**不同解码层级/step 使用不同的活跃 beam 数量**，例如：

```text
Prefill → W=128 → W=64 → W=32 → 输出
```

这里的“层级”不是 Transformer 内部的第 1～N 层。同一次模型 forward 中，各 Transformer 层的 token/batch 维必须一致，不能在中间层随意改变 BW。BW 的变化应发生在一次完整 forward 结束后的 Beam Select 阶段。

## 2. 当前 xAttention 的基本假设

当前实现主要按固定 BW 设计：

```text
输入 token       [B × W]
Unshared KV      [B, W, KVHeads, MaxSteps, HeadDim]
Beam scores      [B, W]
Sequence         [B, W, MaxSteps]
```

每个 decode step 的流程为：

1. `cache_unshared_kv` 写入当前所有 beam 的 KV；
2. `x_attention` 读取 Shared Prompt KV 和 Unshared Beam KV；
3. `beam_search_group` 选择下一步 beam；
4. `select_unshared_kv` 按 parent beam 重排历史 KV。

因此，BW 不只是 Beam Search 的一个参数，还参与 Attention 输入形状、KV 布局、TopK、Graph shape 和 parent index 计算。

## 3. 动态 BW 的主要影响

### 3.1 张量形状变化

当 BW 从 `W_old` 变为 `W_new` 时，下一次 forward 的有效 token 数从：

```text
B × W_old → B × W_new
```

以下张量都需要重新切片或更新：

- input IDs、positions 和 logits indices；
- beam scores、sequence、group token count；
- Unshared KV 的有效 beam 区域；
- Attention metadata 和 Graph runtime shape。

如果只修改 `num_beams` 而没有同步这些张量，会出现 reshape 错误、beam 串组、越界或错误读取 KV。

### 3.2 KV Cache 管理

当前 Unshared KV 按最大 BW 预分配，因此 BW 缩小时可以继续使用前 `W_new` 个 slot，不必重新分配显存。

真正困难的是 parent beam 更新：

- `W_new < W_old`：需要从旧 beam 中筛选并压缩 KV；
- `W_new > W_old`：多个新 beam 会共享同一个 parent，需要复制对应历史 KV；
- 不同 request 使用不同 BW：现有规则的 `[B, W, ...]` 矩形布局难以表达。

短期可以继续使用 `select_unshared_kv` 做物理重排，但动态 BW 越频繁，KV 搬运成本越明显。长期更适合保存 `parent index / BeamPath`，由 xAttention 逻辑寻址历史 KV，避免每一步复制全部历史。

### 3.3 xAttention 算子

xAttention 需要知道当前有效 BW，不能只从预分配 KV 的 `shape[1]` 推断，因为该维度可能是 `max_beam_width`。

建议显式增加或确认以下输入：

```text
active_beam_width
active_token_count
beam/request mapping
```

同时需要确认底层自定义算子是否：

- 只支持固定或对齐后的 BW；
- 支持不同 `B × W` shape；
- 支持重复 parent 和 BW 收缩；
- 能在 ACL Graph replay 时更新有效 BW。

当前仓库只有 Python wrapper，最终能支持哪些 BW 仍取决于外部 xllm 自定义算子的实现。

### 3.4 Beam TopK

需要区分三个概念：

```text
current_width    当前活跃 beam 数
candidate_top_k  每个 beam 产生的候选数
next_width       下一层级保留的 beam 数
```

Beam Select 应从 `current_width × candidate_top_k` 个候选中，全局选出 `next_width` 个结果。

当前实现通常默认 `candidate_top_k = current_width = next_width`。动态 BW 适配时需要修改 `beam_search_group` 的接口和输出 shape，不能继续只传一个 `top_k` 表示所有含义。

### 3.5 ACL Graph

ACL Graph 的运行 shape 通常与 `B × W` 绑定。BW 每次变化都可能需要不同的 Graph。

推荐使用有限的 BW bucket，例如：

```text
8、16、32、64、128
```

实际 BW 向上取整到最近 bucket，通过 mask 屏蔽无效 beam。这样可以控制 Graph 数量，但会引入一定的冗余计算。

Graph cache key 至少应包含：

```text
(batch_bucket, beam_bucket, decode_step_bucket)
```

不能只使用 `B × W`，因为 `B=1,W=128` 与 `B=2,W=64` 的 Beam 分组语义不同。

### 3.6 Scheduler 和多请求

最简单的第一阶段实现是：

- 同一个 batch 内所有请求使用相同 BW；
- 不同 decode 层级允许切换 BW；
- 不同 BW 的请求分到不同 bucket。

如果要求同一 batch 内每个请求拥有不同 BW，则需要引入：

- 每请求 `beam_widths[B]`；
- beam offset/prefix sum；
- ragged token 和 KV mapping；
- 变长 TopK 输出。

这会同时修改 Scheduler、ModelRunner、xAttention 和 Beam 算子，复杂度明显更高，不建议作为第一版目标。

## 4. 实现难度评估

| 目标 | 难度 | 主要原因 |
|---|---|---|
| 固定几个 BW，按 step 切换，同批同宽 | 中 | 需要重切片状态、修改 TopK 语义并准备多套 Graph |
| 任意 BW，使用 bucket/padding | 中高 | 需要 mask、Graph 管理及算子动态有效宽度支持 |
| 同批不同请求使用不同 BW | 高 | 需要 ragged layout 和完整的 request/beam mapping |
| 同一次 Transformer forward 的层间改变 BW | 不建议 | 会破坏 hidden state、残差连接和 KV 的 batch 维一致性 |
| 使用 BeamPath 完全避免 KV 物理重排 | 高 | 需要重新设计 KV 数据模型和 xAttention 寻址方式 |

## 5. 推荐实施方案

建议分两阶段完成。

### 第一阶段：可落地版本

1. 配置每个 decode 层级的 `beam_width_schedule`，例如 `[128, 64, 32]`；
2. 一个 batch 内保持相同 BW；
3. BufferPool 按最大 BW 预分配，运行时只使用有效 slice；
4. 将 Beam Select 改为显式接收 `current_width`、`candidate_top_k` 和 `next_width`；
5. 为常用 BW 建立 ACL Graph bucket；
6. 增加 BW 切换前后的 shape、parent index 和 KV 一致性校验。

### 第二阶段：性能优化

1. 使用 active beam mask 减少 Graph 数量；
2. 避免清零和复制未使用的最大 BW 区域；
3. 将物理 KV 重排逐步改为 BeamPath/parent index 逻辑寻址；
4. 根据实际收益再考虑同批不同 BW。

## 6. 总结

基于当前 xAttention 实现，动态 BW 是可以实现的，但它不是简单地逐步修改 `num_beams`。核心修改集中在：

```text
Beam Select 的宽度语义
+ Beam 状态和 KV 的缩放/重排
+ xAttention 的有效 beam mapping
+ ACL Graph 的多 shape 管理
+ Scheduler 的同宽分桶
```

推荐先支持“**解码层级间动态 BW、同 batch 同 BW、有限 bucket**”。这一方案改动可控，也能覆盖推荐模型中常见的逐层收缩 Beam Search。若进一步追求任意 BW 或同批不同 BW，最好先将当前的 KV 物理重排演进为 BeamPath 逻辑寻址。
