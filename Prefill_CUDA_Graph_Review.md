# PR #332：纯预填充（Prefill）CUDA Graph 全图捕获与重放

---

## 1. 背景与目的

本 PR 为 **NVIDIA GPU** 和 **Ascend NPU** 引入纯预填充（pure prefill）工作负载的**全模型 CUDA Graph 捕获与重放**机制。

**核心目标**：

- 减少短/中长度 prompt 的 kernel launch 与 host dispatch 开销。
- 保留现有 eager 路径作为 fallback，用于不支持或收益为负的 shape。
- 优化仅在满足条件时触发：完整 batch 处于 prefill 阶段、存在兼容 graph、padding 在阈值内。

---

## 2. 实现概要

### 2.1 架构设计

| 组件 | 说明 |
| --- | --- |
| **平台无关运行器** | 提供可复用的静态缓冲区、bucket 匹配、padding 限制、KV 长度边界检查、eager fallback。 |
| **CUDA 实现** | 按 token bucket 捕获单请求全 prefill graph；重放前刷新 input IDs、positions、slot mappings、block tables、attention metadata。 |
| **Ascend 实现** | 按 bucket 捕获单 ACL graph；重放时更新 per-request task metadata；跨 bucket 共享 graph workspace。 |
| **接入点** | 在 vLLM 现有 graph capture 阶段之后链式执行；仅对符合条件的纯 prefill batch 拦截 model forward。 |
| **注意力元数据快速路径** | 纯 prefill 场景跳过 grouped Beam metadata 构造，减少 CPU 开销。 |

### 2.2 容错与 Fallback

以下情况自动回退到 eager 执行：

- Graph 捕获失败
- 无匹配 graph（graph miss）
- Padding 超过阈值
- Slot mapping 缺失
- 请求数量不支持（当前仅支持 `num_reqs=1`）
- KV 长度超出捕获边界

---

## 3. 配置方式

### 3.1 默认 Bucket 配置

默认生成 **64 个 bucket**，序列长度从 **128 到 8192**，步长 **128**：

```bash
# 步长（默认 128）
export VLLM_GR_PREFILL_GRAPH_BUCKET_STEP=128

# 最大 bucket（默认 8192）
export VLLM_GR_PREFILL_GRAPH_MAX_BUCKET=8192
```

### 3.2 开关控制

```bash
# 关闭 prefill graph（完全使用 eager）
export VLLM_ENABLE_PREFILL_CUDAGRAPH=0
```

### 3.3 运行时匹配策略

运行器选择**最小满足条件的 bucket**，刷新其地址稳定的缓冲区后重放 graph。若输入超过配置上限，或 padding 代价过高，则回退 eager。

---

## 4. 性能数据

### 4.1 单长度对比（L20 / A100）

| 硬件 | 输入长度 | 优化前 | 优化后 | 延迟降低 |
| --- | --- | --- | --- | --- |
| NVIDIA L20 | 1K | 57 ms | 37 ms | **35.1%** |
| NVIDIA L20 | 2K | 65 ms | 67 ms | -3.1%（负收益） |
| NVIDIA A100 | 2K | 41 ms | 16 ms | **61.0%** |
| NVIDIA A100 | 4K | 27 ms | 24 ms | 11.1% |
| NVIDIA A100 | 5K | 34 ms | 30 ms | 11.8% |

**趋势**：收益随序列长度增加而递减。Graph replay 主要消除固定 launch/dispatch 开销，而 attention 计算在长序列中占主导。

### 4.2 同版本 A/B 测试（L20，bw128）

| 输入长度 | Prefill Worker (off→on) | E2E (off→on) | 输出一致性 |
| --- | --- | --- | --- |
| 512 | 8.39→4.90 ms (**+41.7%**) | 50.7→47.1 ms | **FAIL** |
| 1,024 | 8.50→6.98 ms (**+17.9%**) | 51.8→49.7 ms | PASS |
| 2,048 | 12.34→11.65 ms (+5.6%) | 57.6→58.3 ms | PASS |
| 4,096 | 23.13→22.76 ms (+1.6%) | 93.2→92.4 ms | PASS |
| 5,120 | 30.34→29.76 ms (+1.9%) | 113.9→112.9 ms | **FAIL** |
| 8,192 | 50.64→50.45 ms (+0.4%) | 183.9→183.3 ms | **FAIL** |
| 10,240 | 66.11→65.83 ms (+0.4%) | 245.0→242.0 ms | **FAIL**（eager fallback） |

> 注：10K 长度因超出最大 bucket（8192），正确回退到 eager。

### 4.3 自动化 AB Benchmark（A-B-B-A，100 样本 × 2）

| 场景 | 指标 | p50 Δ% | p90 Δ% | p99 Δ% | 结论 |
| --- | --- | --- | --- | --- | --- |
| **bw128-in1024** | prefill_hit | **-5.30%** | -8.53% | -7.12% | 🟢 改善 |
| **bw128-in2048** | prefill_hit | **-9.45%** | -7.18% | -13.47% | 🟢 改善 |
|  | decode_hit | — | — | **+6.84%** | 🔴 回归 |
| **bw128-in4096** | prefill_hit | **-13.56%** | -11.41% | -14.06% | 🟢 改善 |
|  | total_beam | -3.05% | — | — | 🟢 改善 |

### 4.4 启用成本

| 项目 | 数值 |
| --- | --- |
| Prefill graph 捕获时间 | ~20.8 s |
| 捕获期间额外设备内存 | ~1.44 GiB |
| 引擎启动时间（off vs on） | 33.4 s vs 59.1 s |
| 捕获 graph 总数 | 64 个 |

---

## 5. 审查发现的问题

### 5.1 🔴 硬编码 `max_model_len=8192` 导致上下文长度回归

**问题**：PR 在 `vllm/gr/arg_utils_gr.py` 中引入 `VLLM_GR_ENGINE_DEFAULTS = {"max_model_len": 8192, ...}`，当调用方未显式传入 `max_model_len` 时强制使用 8192。

**影响**：

- 模型本身支持 `max_position_embeddings=40960`，但 PR 将其限制为 8192。
- 输入长度 ≥ 8192 的请求直接报错：`ValueError: The decoder prompt (length 8193) is longer than the maximum model length of 8192.`
- 10K 长度测试在 head 上完全无法运行，而 baseline 正常。

**建议**：

- 不要硬编码 `max_model_len`。
- 继承模型自身的 context length，或仅在不会缩小模型容量时应用默认值。
- Graph bucket 上限应与模型上下文限制解耦。

### 5.2 🔴 Eager Fallback 路径存在 IndexError

**问题**：当静态缓冲区分配失败时，初始化会清空 `self.buckets`。但后续符合条件的请求进入 `match_bucket()` 时，会访问 `self.buckets[-1]`，引发 `IndexError`。

**建议**：

- 在 `match_bucket()` 中增加空 bucket 列表检查，返回 `None` 以触发 fallback。
- 扩展分配失败的单测，覆盖初始化后的 dispatch/fallback 路径。

### 5.3 🟡 输出一致性未完全通过

| 长度 | 状态 | 说明 |
| --- | --- | --- |
| 512 | FAIL | Ranked token IDs 不一致，但 scores 匹配。可能是 tie-breaking 行为差异，尚未与实现错误隔离。 |
| 1K / 2K / 4K | PASS | 完全匹配 |
| 5K / 8K | FAIL | 即使 graph off 时重复相同请求也会产生变化输出，说明存在现有不稳定性，非 graph replay 独有。 |
| 10K | N/A | 正确回退 eager |

**建议**：

- 区分 tie 行为、现有不稳定性与 replay 引入的缺陷。
- 在声称"正确性保持的收益"前，先解决 512/5K/8K 的不一致问题。

### 5.4 🟡 缺少可复现基准脚本

审查要求提供：

- 可触发 graph replay 的 eligible workload 脚本
- 同版本 head 的 `VLLM_ENABLE_PREFILL_CUDAGRAPH=0/1` 对比
- 精确的 revision、硬件/软件版本、模型/tokenizer、prompt 构造、seed、精度、beam width、并发度
- Warmup 与 cache reset 流程、原始 per-request 测量值、中位数/方差
- 明确 "prefill latency" 的定义边界（GPU 执行 / host+device forward / 前端等待）

---

## 7. 结论

本 PR 在短/中长度 prefill 场景下确实带来了可测量的延迟降低（尤其在 1K~2K 范围），且通过 eager fallback 保持了兼容性。但存在以下**必须后续修复**的缺陷：

- **硬编码 `max_model_len=8192`** 导致长输入被拒绝，属于功能回归。
- **fallback 路径的 IndexError** 可能在生产环境触发崩溃。
- **部分长度输出不一致** 需进一步根因分析。

建议在合并后尽快提交 follow-up PR 解决上述问题。
