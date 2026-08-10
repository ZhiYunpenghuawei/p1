# [RFC][Task 2] 增量解码 Attention 计算路径

- Parent RFC：#249
- Related design：#248
- Sibling RFC：Task 1 / #250
- 范围：增量 Beam Search 的 Attention 计算层
- 日期：2026-08-10
- 代码基线：当前 `decode_graph` 分支
- 状态：设计评审中

## 1. 一句话说明

Task 2 要做的事情是：

> 在增量 Beam Search 的每个 Attention 层中，把当前 Step 新生成的 K/V 写入该层的固定 Pool；Prefix 继续从 Paged KV Cache 读取，Suffix 改为从 Pool 读取；分别计算 Prefix/Suffix Attention 后，用 LSE Merge 得到与现有路径等价的输出。

Task 2 不负责 Beam TopK、Parent 分组、Unshared KV 重排或 Session 生命周期。这些分别属于现有 decode 算子、Task 1 和 Task 3。

## 2. 背景与当前问题

### 2.1 当前 Legacy 路径

当前 `BeamAttentionImpl.forward` 的执行流程是：

```text
当前层产生 Q/K/V
    -> reshape_and_cache：把 K/V 写入 Paged KV Cache
    -> Prefix Attention：从 Paged KV Cache 读取共享 Prefix
    -> extract_suffix_kv：从 Paged KV Cache Gather 每条 Beam 的 Suffix
    -> Suffix Attention
    -> LSE Merge
```

在第 `t` 个增量 Step，每条 Beam 的 Suffix 长度为 `t + 1`。Legacy 路径每一步都会重新 Gather 前面所有 Step 的 Suffix K/V，已经 Gather 过的数据会被重复搬运。

### 2.2 Task 2 的目标路径

```text
当前层产生 Q/K/V
    -> cache_unshared_kv：只把当前 Step K/V 写入当前层 Pool
    -> Prefix Attention：仍从 Paged KV Cache 读取共享 Prefix
    -> Pool 有效区复制到预分配 Suffix Buffer
    -> Suffix Attention：从连续 Suffix Buffer 读取
    -> LSE Merge
```

核心变化只有两点：

1. 当前 Beam Suffix K/V 不再写入 Paged KV Cache，而是写入 Task 1 的 Pool。
2. 增量路径不再调用 `extract_suffix_kv`，而是从 Pool 的有效区准备连续 Suffix K/V。

Attention 的数学语义、Legacy 路径和现有 Paged Prefix Cache 均保持不变。

## 3. Task 1 当前已经提供什么

Task 1 是 Task 2 的数据层依赖，当前已有或正在提交以下工作。

### 3.1 固定地址 BeamAttentionPool

每个本地 Attention 层拥有一对固定地址 K/V Pool：

```text
[1, max_beams, local_num_kv_heads, max_decode_steps, head_dim]
```

已实现的 `BeamAttentionPool` 负责：

- 在初始化时一次性申请 `unshared_key` 和 `unshared_value`；
- 使用 `active_key(beam_width)` / `active_value(beam_width)` 返回有效 Beam View；
- `reset()` 原地清零，不改变地址；
- `nbytes()` 统计 K/V 总显存；
- 校验容量、dtype 和 device；
- Pool 地址在多个 Step 和顺序请求之间保持不变。

Task 2 通过 canonical `layer_name` 获取当前层 Pool，不自行创建或替换 Pool。

### 3.2 通用 OneRec Decode 算子

Task 1 相关 PR 已提供纯 PyTorch GPU 实现：

- `cache_unshared_kv_torch`
- `select_unshared_kv`
- `beam_search_group`
- `beam_search_rec_final_select`

当前实现语义如下：

- `cache_unshared_kv_torch`：把当前 Step K/V 原地写入 5-D Cache 的指定 Block 和时间位置；
- `select_unshared_kv`：先对目标 K/V 做 `clone()` 快照，再按 Parent `gather` 并写回，因此当前实现有临时 K/V 开销，但可以正确处理覆盖问题；
- `beam_search_group`：对全部 `W x W` 候选按累计分数做稳定降序 TopK，再按 Parent 稳定分组，并生成 `group_token_num` 前缀；
- `beam_search_rec_final_select`：对最终候选做稳定 TopK，分数相同时保持较小 flat index 优先；
- 通用算子保持 GPU/NPU 公共 ABI，并已考虑 CUDA Graph 中避免 D2H 的要求。

这些通用算子中的 TopK、Parent 分组和 KV 重排不属于 Task 2。Task 2 只消费 Task 1 面向 Session 的 `cache_unshared_kv(..., step: int)` 接口。

### 3.3 Task 1 仍需完成或合入的接口

Task 2 最终集成依赖 Task 1 冻结的以下能力：

```python
class BeamSearchSession:
    suffix_k_buf: torch.Tensor
    suffix_v_buf: torch.Tensor

    @property
    def is_active(self) -> bool: ...

    @property
    def beam_width(self) -> int: ...

    @property
    def current_step(self) -> int: ...

    @property
    def prefix_len(self) -> int: ...

    def pool_for_layer(self, layer_name: str) -> BeamAttentionPool: ...
```

以及 Task 1 reference facade：

```python
def cache_unshared_kv(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    step: int,
) -> None: ...
```

Task 1 的 `BeamSearchSession` 生命周期、`prepare_next_step(parent)` 和 `select_unshared_kv` 由 Task 1/Task 3 管理。Task 2 不推进 Step，也不重排 Parent。

## 4. Task 2 工作范围

Task 2 总体分为三块。

### 4.1 Incremental Metadata

扩展 `BeamAttentionMetadata`，增加明确的增量模式字段：

```python
is_incremental_decode: bool = False
beam_session: BeamSearchSession | None = None
incremental_suffix_kv_len: int = 0
```

在 `BeamAttentionMetadataBuilder` 中保留现有 Legacy 算法不变，并增加独立的 Incremental 分支。

增量分支固定满足：

- MVP 一次只有一个逻辑 Beam 请求；
- 每条 Beam 当前只有一个 Query Token；
- Query 数量等于 `beam_width`；
- Prefix 长度等于 `session.prefix_len`；
- Suffix 长度 `L = session.current_step + 1`；
- `suffix_total_tokens = beam_width * L`；
- 不再生成 `suffix_slot_mapping_out`；
- Scheduler 的 `decode_step` 只用于和 `session.current_step` 校验，不能覆盖 Session 状态。

模式必须通过 `is_incremental_decode` 显式选择，不能通过 Tensor Shape、Pool 是否存在或 Session 是否为空来猜测。

### 4.2 Forward 显式分流

在 `BeamAttentionImpl.forward` 中增加清晰的 Legacy/Incremental 分支。

Legacy 分支保持当前行为：

```text
reshape_and_cache -> cascade_attention
```

Incremental 分支执行：

```text
1. 校验 Session、Step、Beam Width、dtype、device、平台和容量
2. pool = session.pool_for_layer(layer.layer_name)
3. cache_unshared_kv(pool, current K/V, step=session.current_step)
4. 不把当前 Beam Suffix K/V 写入 Paged KV Cache
5. 调用 cascade_attention_incremental
```

Task 2 在单层 Forward 中不得调用：

- `begin_session()`
- `prepare_next_step(parent)`
- `end_session()`
- `select_unshared_kv()`
- Beam TopK/Group 算子

原因是单个 Attention 层不知道所有层是否已经执行完成，也不拥有 Parent 和最终 Step 决策。

### 4.3 CUDA Pool-Slice Cascade

在 CUDA 后端新增独立入口：

```python
def cascade_attention_incremental(...): ...
```

它负责三个阶段。

第一阶段：Prefix Attention。

- Q 为当前每条 Beam 的一个 Token；
- K/V 从现有 Paged KV Cache 读取；
- 有效 Prefix 长度为 `session.prefix_len`；
- 保持现有 scale、window、ALiBi、softcap、FA version 和 LSE 参数语义。

第二阶段：Suffix Attention。

当前层 Pool 的有效区为：

```text
pool[0, :W, :, :L, :]
```

其逻辑 Shape 为：

```text
[W, H, L, D]
```

FlashAttention varlen 需要 Beam-major、Step-minor 的连续 TND Buffer：

```text
suffix[(beam * L) + step, head, dim]
```

因此把 Pool 有效区复制到 Task 1 预分配的：

```text
session.suffix_k_buf[:W * L]
session.suffix_v_buf[:W * L]
```

禁止通过可能隐式分配的 `permute().reshape()` 获得连续副本；目标必须是固定 Buffer View，并显式 `copy_` 或使用等价的无额外 Storage 实现。

第三阶段：LSE Merge。

- Prefix 和 Suffix 分别返回 output 与 LSE；
- 保留现有 `merge_attn_states` 数学语义；
- 保留 FA2/GQA 的 LSE Layout 修正；
- 最终结果写入调用方提供的 `output`；
- 不在增量热路径新建 `torch.empty_like(beam_query)` Merge Buffer。

## 5. 执行时序

假设当前是 Step `t`，Task 2 在每个本地 Attention 层中的时序为：

```text
Task 3 已经开始 Session，session.current_step = t

Layer 0 Forward:
    写 Layer 0 Pool[..., t, :]
    Prefix Attention
    Layer 0 Pool[...,:t+1,:] -> 固定 Suffix Buffer
    Suffix Attention + LSE Merge

Layer 1 Forward:
    写 Layer 1 Pool[..., t, :]
    Prefix Attention
    Layer 1 Pool[...,:t+1,:] -> 同一个固定 Suffix Buffer
    Suffix Attention + LSE Merge

...

所有层 Forward 完成后：
    Task 3 得到下一轮 Parent
    Task 3 调用 session.prepare_next_step(parent)
    Task 1 重排所有层 Pool，并把 Step 推进到 t + 1
```

因此一份 Suffix Buffer 可以按层复用；但必须保证同一 Stream 中上一层对 Buffer 的读取完成后，下一层才能覆盖它。多 Stream 并发不在 MVP 中。

## 6. 冻结的跨任务契约

| 项目 | 冻结约定 |
| --- | --- |
| 模式开关 | 只使用 `is_incremental_decode` 显式分流 |
| Step 权威 | `session.current_step: int` |
| Prefix 权威 | `session.prefix_len: int` |
| Layer 定位 | `session.pool_for_layer(layer.layer_name)` |
| Pool 写入 | 当前层、当前 Step、Active Beam |
| Pool 读取 | 当前层、Active Beam、`[0, current_step]` |
| Parent 重排 | Forward 完成后由 Task 1/Task 3 执行 |
| TopK/Group | 不属于 Task 2 |
| Paged Cache | 增量路径只读取共享 Prefix，不写 Beam Suffix |
| 错误处理 | Active Incremental Session 中校验失败必须报错，不允许中途静默切回 Legacy |
| Legacy | Builder、Forward、CUDA/NPU Cascade 现有语义不变 |

Task 3 需要向本地执行上下文提供：

```python
{
    "is_incremental_decode": True,
    "beam_width": int,
    "beam_decode_steps": int,
    "prefix_len": int,
    "cache_len": int,
    "decode_step": int,
}
```

以及当前 ModelRunner 本地持有的 `BeamSearchSession` 引用。Session/Tensor 引用不得放进需要跨进程序列化的 Scheduler 数据。

## 7. MVP 范围与 Gate

首版仅支持：

- CUDA；
- `max_batch = 1`；
- 每个 ModelRunner 最多一个 Active Incremental Session；
- 每条 Beam 每个 Forward 恰好一个 Token；
- FP16/BF16，且 Model K/V、Pool、Suffix Buffer dtype/device 完全一致；
- 非量化 Paged Prefix KV Cache；
- 固定 `beam_width` 和固定最大 decode step 容量；
- 单 Stream 顺序执行。

以下情况必须在 Session 激活前 Gate 到 Legacy：

- NPU Incremental Attention；
- FP8/量化 KV；
- KV Sharing；
- 多逻辑请求混批或 `max_batch > 1`；
- 多 Active Session；
- Regrow/Recombination；
- 超过 Pool 容量；
- CUDA Graph Capture/Replay；
- 多 Stream 复用同一个 Suffix Buffer。

Task 2 合入后 Feature Gate 默认关闭，不能单独改变生产执行路径。

## 8. 非目标

Task 2 不实现：

- Pool、Session 或 Persistent Buffer 的创建和生命周期；
- Unshared KV Parent 重排算法；
- 无缓冲区原地重排优化；
- Beam TopK、稳定排序、Parent 分组和最终序列选择；
- Scheduler Timeline、输入重映射或 Host Beam 决策；
- NPU Incremental Cascade；
- 自定义 CUDA/Triton Pool Attention Kernel；
- CUDA Graph；
- 端到端默认启用和性能承诺。

其中，Task 1 当前 `select_unshared_kv` 的 `clone + gather` 临时开销属于后续数据层优化，不应扩大 Task 2 范围。

## 9. 测试与验收

### 9.1 Metadata 测试

至少覆盖：

- `W = 1`、普通宽度、最大宽度；
- `current_step = 0` 和最大合法 Step；
- Prefix 跨一个或多个 Cache Block；
- Query 数量固定为 `W`；
- `L = current_step + 1`；
- `suffix_total_tokens = W * L`；
- 不生成 `suffix_slot_mapping_out`；
- Session 缺失/未激活、Step/W/Prefix 不一致、混批、容量越界时失败；
- Legacy Metadata 输出保持不变。

### 9.2 Forward 路由测试

通过 Fake Session/Pool 和 Mock Platform 验证：

- Legacy 只调用原 `reshape_and_cache + cascade_attention`；
- Incremental 不调用 Paged `reshape_and_cache` 写 Beam Suffix；
- Incremental 每层恰好调用一次 `cache_unshared_kv`；
- 使用原始 `layer.layer_name` 获取 Pool；
- Step 使用 Host `session.current_step`，不调用 Device Tensor `.item()`；
- Scale、Window、ALiBi、Sinks、Softcap 和 FA Version 完整透传；
- Task 2 不修改 Session 生命周期或 Step。

### 9.3 Pool 到连续 Buffer 测试

构造可追踪数据：

```text
pool[b, h, t, d] = encode(b, h, t, d)
```

验证：

```text
suffix[(b * L) + t, h, d] == pool[0, b, h, t, d]
```

覆盖 `W/L` 的最小、部分容量和最大容量，以及 FP16/BF16。Inactive Beam 和 Future Step 不得复制到有效 Buffer。

### 9.4 CUDA 等价测试

使用相同 Q/K/V 对比：

```text
Baseline：Paged Cache + extract_suffix_kv + Legacy Cascade
Candidate：Pool 写入 + 固定 Buffer Copy + Incremental Cascade
```

逐层、逐 Step 比较：

- Pool 有效历史；
- 连续 Suffix Buffer 顺序；
- Prefix output/LSE；
- Suffix output/LSE；
- Merge 最终输出；
- Task 1 Parent 重排后的下一 Step 输出。

Parent 至少覆盖 identity、reverse 和 duplicate，例如：

```text
[0, 1, 2, 3]
[3, 2, 1, 0]
[3, 1, 1, 1]
```

### 9.5 分配与回归测试

验证增量热路径：

- 不调用 `extract_suffix_kv`；
- Pool-to-Buffer 不产生新的连续 K/V Storage；
- 不创建 Merge `torch.empty_like`；
- Pool 和 Suffix Buffer `data_ptr()` 跨层、跨 Step 不变。

Legacy 回归至少覆盖：

```text
tests/test_beam_attention_metadata.py
tests/test_beam_attention_platform_execution.py
tests/kernels/attention/test_cascade_beam_attn_gpu.py
tests/kernels/attention/test_cascade_beam_attn_npu.py
```

## 10. PR 拆分

Task 2 建议拆成三个实现 PR。PR 1 和 PR 2 可以并行，PR 3 负责最终接线。

### PR 1：Incremental BeamAttentionMetadata

范围：

- `BeamAttentionMetadata` 新字段；
- 显式模式分类和 MVP Guard；
- `_build_beam_mappings_incremental`；
- `_create_beam_tensors_incremental`；
- Metadata CPU 单元测试；
- Legacy Metadata 回归测试。

评审重点：

- Legacy Builder 算法不变；
- 每 Beam 一个 Query Token；
- Suffix 长度、`cu_seqlens` 和 Prefix Block Table 正确；
- 不生成 `suffix_slot_mapping_out`；
- Session/Step/Width/Prefix/Batch 校验完整。

依赖：可先使用符合 Task 1 Frozen API 的 Fake Session，不必等待真实 Session 完全合入。

### PR 2：CUDA Pool-Slice Cascade

范围：

- CUDA `cascade_attention_incremental`；
- Pool-to-Suffix-Buffer 布局复制；
- Prefix/Suffix Attention 和 LSE Merge；
- FA2/GQA LSE 修正；
- NPU 明确拒绝 Incremental 的接口保护；
- CUDA 层级等价测试与分配测试。

评审重点：

- Buffer 布局正确；
- 不调用 `extract_suffix_kv`；
- 不通过非连续 `reshape` 隐式分配；
- 不替换 Task 1 Persistent Tensor；
- 输出和 Legacy Baseline 等价。

依赖：可使用 Fake Pool 和预分配 Buffer，独立于真实 Session 生命周期开发。

### PR 3：Forward 与 Session 接线

范围：

- `BeamAttentionImpl.forward` Legacy/Incremental 显式分流；
- `pool_for_layer(layer.layer_name)`；
- 调用 Task 1 `cache_unshared_kv(..., step: int)`；
- 消费 Task 3 提供的本地 Session Context；
- Unsupported Gate；
- Forward Mock、Task 1/2 集成和完整 Legacy 回归测试。

评审重点：

- 增量 Suffix 不写入 Paged Cache；
- 每层写入自己的 Pool；
- Task 2 不推进 Session、不执行 Parent 重排；
- dtype/device/Step/容量校验完整；
- Active Session 错误快速失败；
- Feature Gate 默认关闭。

依赖：Task 1 Pool、reference facade、Session 和 Task 3 最小本地 Context 接口。

依赖关系：

```text
Task 2 PR 1：Metadata --------------------+
                                          \
Task 2 PR 2：CUDA Cascade -----------------> Task 2 PR 3：Forward Integration
                                          /
Task 1：Pool + Facade + Session ----------+

Task 3：Local Session Context ------------+
```

## 11. 完成标准

Task 2 只有在以下条件全部满足后才算完成：

- 三个 PR 均合入；
- Incremental Metadata 数值和 Guard 测试通过；
- 当前层 K/V 正确写入对应 Pool 的当前 Step；
- Prefix 从 Paged KV Cache 读取；
- Suffix 从 Pool 有效区读取；
- Incremental 路径不再调用 `extract_suffix_kv`；
- Prefix/Suffix/LSE Merge 与 Legacy Baseline 等价；
- Pool 和 Suffix Buffer 地址保持不变；
- Task 2 不执行 TopK、Parent 分组、KV 重排或 Session Step 推进；
- CUDA FP16/BF16 测试通过；
- Legacy CUDA/NPU/FP8/KV-Sharing 路径不受影响；
- Task 1/Task 3 交接测试通过；
- Feature Gate 默认关闭。

## 12. 主要风险

| 风险 | 处理方式 |
| --- | --- |
| Pool 布局到 TND Buffer 的顺序错误 | 使用可追踪值逐元素验证 `(beam, step, head, dim)` |
| Session Step 与 Scheduler Step 不一致 | Builder 中比较 Host int，失败时不推进 Session |
| 当前层使用错误 Pool | 只允许 canonical `layer_name` 查找 |
| Active Session 中途回退 Legacy | 禁止静默回退，直接失败并由 Task 3 清理 |
| FA2/GQA LSE Layout 不一致 | 保留现有修正并增加 FA2/FA3 对比测试 |
| 固定 Buffer 被下一层过早覆盖 | MVP 限制同 Stream 顺序执行 |
| Incremental 路径意外产生临时 K/V | Profiler/Memory History 检查分配与地址 |
| Task 1 接口尚未全部合入 | PR 1/2 使用 Fake Contract，PR 3 等待真实接口接线 |

后续如果要支持 CUDA Graph、多 Session、多 Batch、NPU、FP8、KV Sharing、直接读取 5-D Pool 的自定义 Kernel，均应单独设计，不纳入本 RFC 的三个 PR。
