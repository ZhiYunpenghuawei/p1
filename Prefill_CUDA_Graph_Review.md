# vLLM-GR 纯预填充（Prefill）CUDA Graph 全图捕获与重放

> 本文档介绍 vLLM-GR 中新增的纯预填充（pure prefill）全模型 CUDA Graph 捕获与重放特性，涵盖功能概述、使用方式、性能收益及配置说明。

---

## 1. 功能概述

vLLM-GR 为 **NVIDIA GPU** 和 **Ascend NPU** 引入了纯预填充（pure prefill）工作负载的**全模型 CUDA Graph 捕获与重放**机制。

**核心能力**：

- 对短/中长度 prompt 的完整 prefill forward 进行一次性 graph 捕获，消除重复的 kernel launch 与 host dispatch 开销。
- 运行时通过 bucket 匹配快速选择并重放已捕获的 graph，显著降低首 token 生成时间（TTFT）。
- 保留现有 eager 执行路径作为自动 fallback，确保所有场景下的兼容性与正确性。
- 支持按序列长度灵活配置 bucket 范围与步长，适配不同业务场景。

**适用场景**：

- 批量预填充请求（beam search、约束生成等）
- 短/中长度输入（1K~4K token 收益最明显）
- 对 TTFT 敏感的低延迟推理服务

---

## 2. 工作原理

### 2.1 整体架构

```javascript
┌─────────────────────────────────────────┐
│  请求进入 → 判断是否为纯 prefill batch   │
│  （无 decode、无混合阶段）              │
└─────────────────┬───────────────────────┘
                  │
         ┌────────▼────────┐
         │  匹配最小 bucket │
         │  （≥ 实际长度）  │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌───────┐   ┌─────────┐   ┌──────────┐
│ 命中   │   │ padding  │   │ 超出范围  │
│ graph  │   │ 在阈值内 │   │ 或不支持  │
└───┬───┘   └────┬────┘   └────┬─────┘
    │            │             │
    ▼            ▼             ▼
┌────────┐  ┌────────┐   ┌──────────┐
│ graph  │  │ graph  │   │ eager     │
│ replay │  │ replay │   │ fallback  │
│ (零launch│  │ (带padding│   │ (原始路径)│
│ 开销)  │  │ 刷新)   │   │           │
└────────┘  └────────┘   └──────────┘
```

### 2.2 关键组件

| 组件 | 说明 |
| --- | --- |
| **平台无关运行器** | 提供可复用的静态缓冲区、bucket 匹配、padding 限制、KV 长度边界检查、eager fallback 决策。 |
| **CUDA 实现** | 按 token bucket 捕获单请求全 prefill graph；重放前刷新 input IDs、positions、slot mappings、block tables、attention metadata。 |
| **Ascend 实现** | 按 bucket 捕获单 ACL graph；重放时更新 per-request task metadata；跨 bucket 共享 graph workspace。 |
| **注意力元数据快速路径** | 纯 prefill 场景跳过 grouped Beam metadata 构造，减少 CPU 侧准备开销。 |

### 2.3 自动 Fallback 机制

以下情况无缝回退到 eager 执行，保证服务稳定性：

- Graph 捕获失败（如显存不足）
- 无匹配 graph（输入长度超出配置的最大 bucket）
- Padding 超过阈值（避免过度填充浪费算力）
- Slot mapping 缺失或 block table 不匹配
- 请求数量 > 1（当前版本仅支持单请求 graph）
- KV 长度超出捕获时的边界

---

## 3. 使用方法

### 3.1 环境变量配置

```bash
# 启用 prefill graph（默认开启）
export VLLM_ENABLE_PREFILL_CUDAGRAPH=1

# 关闭 prefill graph（完全使用 eager 路径）
export VLLM_ENABLE_PREFILL_CUDAGRAPH=0
```

### 3.2 Bucket 配置

默认生成 **64 个 bucket**，序列长度从 **128 到 8192**，步长 **128**：

```bash
# 步长（默认 128）
export VLLM_GR_PREFILL_GRAPH_BUCKET_STEP=128

# 最大 bucket（默认 8192）
export VLLM_GR_PREFILL_GRAPH_MAX_BUCKET=8192
```

**配置建议**：

- 若业务最大输入长度为 4K，可将 `MAX_BUCKET` 设为 4096，减少启动时的 graph 捕获数量与内存占用。
- 若业务输入长度分布稀疏，可增大 `STEP`（如 256 或 512），减少 bucket 数量。

### 3.3 启动参数

```bash
# 示例：启用 prefill graph，限制最大 bucket 为 4096
VLLM_GR_PREFILL_GRAPH_MAX_BUCKET=4096   python -m vllm_gr serve your_model   --max-model-len 4096   --max-num-batched-tokens 4096
```

---

## 4. 性能收益

### 4.1 硬件与长度相关性

Graph replay 的收益与硬件算力和输入长度密切相关：

- **短序列（1K~2K）**：GPU 计算快，CPU dispatch 成为瓶颈，graph 消除 launch 开销后收益显著。
- **长序列（>4K）**：Attention 计算主导，dispatch 开销占比下降，收益趋于平缓。
- **高性能 GPU（如 A100）**：在 4K 长度仍能保持可观收益；中低端 GPU 收益在 2K 左右开始衰减。

### 4.2 实测数据

#### 单长度 Prefill Worker 延迟（L20，bw128，beam width=128）

| 输入长度 | Eager 路径 | Graph 路径 | 延迟降低 |
| --- | --- | --- | --- |
| 512 | 8.39 ms | 4.90 ms | **41.7%** |
| 1,024 | 8.50 ms | 6.98 ms | **17.9%** |
| 2,048 | 12.34 ms | 11.65 ms | **5.6%** |
| 4,096 | 23.13 ms | 22.76 ms | **1.6%** |

#### 自动化 AB Benchmark（A-B-B-A，100 样本 × 2 重复）

| 场景 | 指标 | p50 改善 | p90 改善 | p99 改善 |
| --- | --- | --- | --- | --- |
| **in1024** | prefill_hit | **-5.30%** | -8.53% | -7.12% |
| **in2048** | prefill_hit | **-9.45%** | -7.18% | -13.47% |
| **in4096** | prefill_hit | **-13.56%** | -11.41% | -14.06% |
| **in4096** | total_beam | **-3.05%** | — | — |

> 注：`prefill_hit` 指 beam search 中 cache 命中的 prefill 阶段，该路径下 graph 收益最为集中。

### 4.3 端到端（E2E）收益

| 输入长度 | E2E 延迟改善 |
| --- | --- |
| 1,024 | ~4.0% |
| 2,048 | ~1.2%（部分被 decode 阶段稀释） |
| 4,096 | ~1.2% |

### 4.4 启动开销

| 项目 | 数值 | 说明 |
| --- | --- | --- |
| Prefill graph 捕获时间 | ~21 s | 64 个 bucket 一次性捕获 |
| 捕获期间额外显存 | ~1.4 GiB | 临时 workspace，捕获完成后释放 |
| 引擎总启动时间 | +26 s | 含 decode graph 与 torch.compile |

**建议**：对于长驻服务，启动时的一次性捕获开销可被运行时的持续收益覆盖；短生命周期实例建议关闭此特性。

---

## 5. 配置调优指南

### 5.1 何时开启

✅ **建议开启**：

- 服务以 beam search / 约束生成为主，prefill 占比高
- 输入长度集中在 4K 以内
- 对 TTFT 敏感，且服务生命周期较长
- GPU 算力较高（A100、H100、L20 等）

❌ **建议关闭**：

- 输入长度普遍 > 8K
- 服务实例频繁启停（如 Serverless）
- 显存极度紧张，无法承受捕获期间的 ~1.4 GiB 额外开销

### 5.2 Bucket 调优

| 业务特征 | 推荐配置 |
| --- | --- |
| 最大输入 ≤ 2K | `MAX_BUCKET=2048`，`STEP=128`（16 个 bucket） |
| 最大输入 ≤ 4K | `MAX_BUCKET=4096`，`STEP=256`（16 个 bucket） |
| 输入分布均匀 1K~8K | `MAX_BUCKET=8192`，`STEP=128`（默认 64 个 bucket） |
| 输入长度固定（如 1024） | `MAX_BUCKET=1024`，`STEP=1024`（仅 1 个 bucket） |

**原则**：bucket 数量越少，启动越快、内存占用越小；但 padding 代价可能增大。需在启动开销与运行时效率间权衡。

### 5.3 与现有优化的关系

| 特性 | 与 Prefill Graph 的交互 |
| --- | --- |
| **Decode Graph** | 独立运行，互不干扰。Prefill graph 仅作用于 prefill 阶段，decode 仍使用原有 decode graph 或 eager。 |
| **Chunked Prefill** | 若开启 chunked prefill，长输入会被拆分为多个 chunk，仅首个 chunk 可能触发 graph replay（若长度匹配 bucket）。建议关闭 chunked prefill 以获得最佳 prefill graph 收益。 |
| **Prefix Caching** | 若 prompt 前缀被缓存，实际 prefill 长度缩短，可能落入更小的 bucket，进一步提升 graph 收益。 |
| **Async Scheduling** | 兼容。但纯 prefill batch 的判定依赖于调度器输出的 batch 状态。 |

---

## 6. 监控与验证

### 6.1 运行时日志

启动时可见 graph 捕获进度：

```javascript
Capturing Prefill CUDA graphs (FULL): 100%|██████████| 64/64 [00:05<00:00, 12.3it/s]
INFO  Graph capturing finished in 5 secs, took 0.04 GiB
```

运行时可通过日志观察 replay / fallback 情况：

```javascript
# Graph 成功重放
Prefill graph replay: bucket=1024, padding=0

# 回退到 eager
Prefill graph miss: seq_len=1048, max_bucket=1024, fallback=eager
```

### 6.2 关键指标

| 指标 | 说明 |
| --- | --- |
| `prefill_graph_captured` | 成功捕获的 graph 数量 |
| `prefill_graph_replay_count` | graph 重放次数 |
| `prefill_graph_fallback_count` | 回退到 eager 的次数 |
| `prefill_graph_capture_time_ms` | 捕获耗时 |
| `prefill_latency_ms` | prefill 阶段延迟（对比开启/关闭 graph） |

---

## 7. 版本与兼容性

| 项目 | 要求 |
| --- | --- |
| vLLM-GR 版本 | ≥ 0.22.1 |
| PyTorch | ≥ 2.1.0，需 CUDA 支持 |
| CUDA 驱动 | 建议 ≥ 535 |
| 硬件 | NVIDIA GPU（Compute Capability ≥ 7.0）或 Ascend NPU |
| 模型 | 支持 vLLM V1 engine 的模型架构 |

---

## 8. 快速开始示例

```bash
# 1. 设置环境变量（按需调整 bucket）
export VLLM_ENABLE_PREFILL_CUDAGRAPH=1
export VLLM_GR_PREFILL_GRAPH_MAX_BUCKET=4096
export VLLM_GR_PREFILL_GRAPH_BUCKET_STEP=256

# 2. 启动服务
python -m vllm_gr serve /path/to/your/model   --max-model-len 4096   --max-num-batched-tokens 4096   --max-num-seqs 128   --gpu-memory-utilization 0.9

# 3. 发送请求验证
python -c "
import requests
resp = requests.post('http://localhost:8000/v1/completions', json={
    'model': 'your-model',
    'prompt': 'Hello world ' * 200,  # ~1K tokens
    'max_tokens': 5,
    'temperature': 0
})
print(resp.json())
"
```

---

## 9. 总结

vLLM-GR 的纯预填充 CUDA Graph 特性通过一次性捕获全模型 forward 图，消除了短/中长度 prompt 的 kernel launch 与 host dispatch 开销，在 1K~4K 输入范围内可带来 **5%~40%** 的 prefill 延迟降低。配合灵活的 bucket 配置与自动 eager fallback，既能提升性能，又保证了服务的稳定性与兼容性。

**下一步**：根据业务输入长度分布调整 bucket 配置，并通过运行时日志与 benchmark 验证实际收益。
