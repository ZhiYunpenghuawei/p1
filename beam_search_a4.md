# Beam Search V1 A4 PR #383 中文代码导读

本文介绍 JiusiServe/vllm-gr `decode_graph` 分支中 PR #383 的实现。阅读基线为：

- `decode_graph`：`0efef35`
- A4 PR #383 merge commit：`543d20f`
- PR 标题：`feat(beam): add persistent GPU beam state and execution primitives`

本文先解释 A4 解决什么问题，再介绍数据和文件，之后沿一次请求的生命周期串起完整执行流程。最后说明它与 A3 的关系以及后续集成仍缺少什么。

## 1. 先用一句话理解 A4

A4 的目标是：模型在 GPU 上产生候选 token 后，Beam 的候选打分、EOS 判断、Top-W 选择、序列更新和 KV 重排都继续在 GPU 上完成，不在每个 Decode step 把 token、score、parent 或计数取回 CPU。

A4 #383 实现的是一个独立 GPU 执行组件，主要入口是：

```python
runner.initialize_from_prefill(token_ids, logprobs, valid_mask)
runner.advance(token_ids, logprobs, valid_mask)
```

它还没有接入生产 ModelRunner。当前可以把它理解成已经实现和充分测试的“GPU Beam 状态机”，但还缺少一层胶水把 A3 调度结果、模型 forward、候选生成和这个状态机连接起来。

## 2. 它要消除的旧流程问题

传统的逐步 Beam 流程通常是：

```text
模型在 GPU 上得到 logits
        ↓
候选结果传到 CPU
        ↓
CPU 计算/读取 Beam 选择结果
        ↓
CPU 构造下一轮 token 和 parent
        ↓
重新发起下一次 Decode
```

这样每两个 Decode 之间存在强制的 host dependency。即使下一轮真正需要的数据都已经在 GPU 上，也必须等待 CPU 看见上一轮结果。

A4 改成：

```text
模型在 GPU 上得到候选
        ↓
GPU 判断 active/completed 候选
        ↓
GPU 选择下一轮 Beam
        ↓
GPU 更新 token、score、parent 和历史
        ↓
GPU 按 parent 重排 KV
        ↓
下一轮模型直接消费 GPU 状态
```

CPU 只在最终结果消费、错误检查或资源释放边界参与，不再控制每个 Beam step。

## 3. 先理解三个维度

### 3.1 Beam width W

`W` 是保留的 active Beam 数，也是每个 active parent 产生的候选数。

- Prefill 只有一个隐式 root parent，因此候选 shape 是 `[1, W]`。
- Decode 有 W 个 parent，每个 parent 有 W 个候选，因此候选 shape 是 `[W, W]`。

例如 W=3：

```text
Prefill candidates:
root → [A, B, C]

Decode candidates:
A → [A1, A2, A3]
B → [B1, B2, B3]
C → [C1, C2, C3]
```

一次 Decode 最多需要在 9 个候选中选择下一轮 3 个 active Beam。

### 3.2 active 和 completed

候选分为两类：

- active candidate：没有结束，可以进入下一轮 Top-W。
- completed candidate：命中 stop token 或达到最大长度，保存起来等待最终输出。

completed candidate 不会继续占用下一轮 active Beam，但不能被丢弃。尤其是一个 EOS 候选即使分数没有进入 active Top-W，也仍可能进入最终结果，所以 A4 会保存所有有效的 completed candidates。

### 3.3 parent mapping

下一轮 Beam 可能来自任意上一轮 parent，例如：

```text
new beam 0 ← old beam 1
new beam 1 ← old beam 1
new beam 2 ← old beam 0
```

这时 token 序列和 KV 都必须使用同一份 parent mapping 更新。特别是两个 child 同时选择 parent 1，KV 重排必须支持重复 parent，不能直接做会相互覆盖的原地复制。

## 4. A4 的资源和数据所有权

A4 没有重新创建一套 Session、KV Pool 或 Beam selection 系统，而是复用现有资源。

```text
BeamAttentionContext
├── BeamDecisionBuffers
│   ├── sequence
│   └── beam_scores
├── KV pools
├── KV reorder auxiliary buffers
└── per-layer KV pointer tables

BeamSearchSession
├── session_id
└── KV slot

BeamPostprocessWorkspace
├── selection scratch
├── selected token/parent/score
├── next_sequence
└── A4 BeamStateScratch（按需创建）

BeamSearchState
├── 复用 sequence 和 beam_scores
├── 当前 input/parent/active 状态
├── GPU counters
├── append-only history
├── completed records
└── sticky error

BeamSearchRunner
└── 绑定以上资源并执行一步 GPU Beam 更新
```

### 4.1 为什么区分 KV slot 和 state index

Session 的 `slot` 表示它使用哪个 KV Pool 槽位；`state_index` 表示它使用 `BeamSearchState` 的哪一行。

```text
KV slot       → KV 存储位置
state_index   → Beam 状态存储位置
workspace row → 当前实现固定使用 row 0
```

当前 A4 只允许一个 active runner，但代码不强制 `KV slot == state_index`。测试中覆盖了 state row 1 配合 KV slot 0，以避免未来集成时错误地把两个索引当成同一个概念。

### 4.2 哪些是持久状态，哪些是临时状态

持久状态跨 Decode step 保存：

- active sequence；
- active cumulative score；
- 下一轮 input token 和 parent；
- active mask 和 sequence length；
- generated/decode/completed 计数；
- token/parent history；
- completed records；
- initialized/finished/error 状态。

临时 Workspace 只服务当前一步：

- candidate cumulative score；
- terminal mask/rank；
- selection mask；
- Top-W selection scratch；
- staged next sequence；
- `execute`、`reorder` 等设备侧 latch。

## 5. 建议的源码阅读顺序

建议不要从 Triton kernel 开始读。更容易理解的顺序是：

1. `vllm_gr/v1/beam/beam_state.py`
2. `vllm_gr/v1/worker/beam_search_runner.py`
3. `vllm_gr/ops/cuda/beam_state.py`
4. `vllm_gr/ops/cuda/beam_selection.py`
5. `vllm_gr/ops/cuda/beam_kv_reorder.py`
6. `beam_decision.py`、`beam_attention_context.py`、`beam_search_session.py`

前两个文件定义“数据是什么”和“一步做什么”；后面的 kernel 文件解释“这一步怎样在 GPU 上实现”。

## 6. `beam_state.py`：一次请求的 GPU 状态

核心类是 `BeamSearchState`。

### 6.1 `BeamSearchState.__init__()`

构造函数完成四件事。

第一，校验固定配置：

- `beam_width` 和 `max_new_tokens` 必须是正整数；
- Beam width 不能超过已有 decision buffer 和 CUDA selection 容量；
- token history 不能超过 sequence buffer 的物理长度；
- stop token 必须是唯一、非负、可放入 int32 的整数。

第二，复用已有 decision storage：

```python
self.sequence = decision.sequence[:, :beam_width]
self.beam_scores = decision.beam_scores[:, :beam_width]
```

这两个 Tensor 由 `BeamDecisionBuffers` 所有，`BeamSearchState` 只是建立 view。

第三，分配新的固定地址 Tensor：

```text
input_ids              [Bmax, W]
parent_ids             [Bmax, W]
active_mask            [Bmax, W]
sequence_lengths       [Bmax, W]

num_generated_tokens   [Bmax]
num_decode_steps       [Bmax]
num_active_beams       [Bmax]
num_completed          [Bmax]
is_initialized         [Bmax]
is_finished            [Bmax]
error_code             [Bmax]

history_token_ids      [Bmax, T, W]
history_parent_ids     [Bmax, T, W]

completed_*            [Bmax, C]
```

第四，为每个 state row 执行一次 `reset()`。

### 6.2 completed capacity

A4 使用保守容量：

```text
C = W + (T - 1) × W²
```

原因是：

- Prefill 最多有 W 个完成候选；
- 后续 T-1 轮 Decode，每轮最多有 W² 个完成候选。

这样不需要在最终语义确定前截断 EOS 候选，但代价是 completed storage 随 W² 增长。W 很大时，这是需要重点关注的显存开销。

### 6.3 `owned_tensors()` 和 `device_memory_bytes`

`owned_tensors()` 只返回 State 自己分配的 Tensor，不包含复用的 `sequence` 和 `beam_scores`。

`device_memory_bytes` 据此计算 A4 State 新增的有效 payload 字节数，避免重复统计已有 decision storage。

### 6.4 `reset(state_index)`

重置一行运行状态，但保留请求的固定配置：

- `max_new_tokens_tensor`
- `stop_token_ids`
- `num_stop_tokens`
- `ignore_stop_tokens`

分数 Tensor 填充为 `-inf`，其他运行状态清零。Tensor 地址不变，因此同一资源绑定可以被 CUDA Graph 重新 replay。

`reset()` 只清状态，不等待 stream，也不释放 KV slot。调用顺序由上层负责。

## 7. `beam_search_runner.py`：A4 的执行入口

核心类是 `BeamSearchRunner`。

### 7.1 `BeamSearchRunner.__init__()`

构造函数把 Session、State、Workspace 和 KV 资源绑定为一个执行单元。

它会校验：

- State 必须位于 CUDA；
- State 必须复用当前 Context 的 decision buffers；
- Workspace 的 beam width、top-k、最大 Decode 步数和 device 必须匹配；
- KV Pool 的层数、dtype、shape 和 capacity 必须一致；
- 每层 KV pointer table 和 aux buffer 必须已经准备好；
- `state_index` 必须处于物理容量内。

然后它只建立一次固定 view 和 kernel 参数：

- 当前 State row 的 view；
- Workspace row 0 的 view；
- selected token/parent/score buffer；
- KV Pool slot offset；
- 每层地址、stride 和 launch grid。

这一步的目的不仅是减少 Python 开销，也是在 CUDA Graph capture 后保持所有资源地址稳定。

### 7.2 `initialize_from_prefill()`

接收：

```text
token_ids  int32   [1, W]
logprobs   float32 [1, W]
valid_mask bool    [1, W]
```

它调用 `_advance(..., prefill=True)`。Prefill 只有一个 root parent，因此没有 KV parent reorder。

### 7.3 `advance()`

接收：

```text
token_ids  int32   [W, W]
logprobs   float32 [W, W]
valid_mask bool    [W, W]
```

它调用 `_advance(..., prefill=False)`。

在调用之前，上层必须已经把本轮模型产生的 Decode KV 写入 KV Pool 的 `num_decode_steps` 位置。A4 随后按本轮选择出的 parent mapping 对这些 KV 做一次重排。

### 7.4 `_advance()`

这是 A4 最重要的函数。它不把结果读到 CPU，而是在当前 CUDA stream 上依次 enqueue：

```text
准备并分类 candidates
        ↓
设备侧校验
        ↓
选择下一轮 Top-W active candidates
        ↓
保存全部 completed candidates
        ↓
构造并提交下一轮 active sequence
        ↓
按 parent mapping 重排 KV
        ↓
最后提交 counters 和 finished 状态
```

后文会逐步展开这个调用链。

### 7.5 `reset()`

调用 `BeamSearchState.reset(state_index)`，用于相同固定配置和资源地址下的下一次请求。

如果最终结果由另一条 CUDA stream 消费，上层必须先让当前 stream 等待 consumer event，再调用 reset。

### 7.6 `end_session(consumer_done=None)`

这是资源释放边界，不属于 steady path，也不允许放进 graph capture。

执行顺序：

1. 如果传入另一条 stream 的 `consumer_done` event，当前 stream 先等待它。
2. 在当前 stream 记录一个新的完成 event。
3. 同步等待这个 event。
4. 调用 `BeamSearchSession.end_session()` 归还 KV slot。
5. 将 Runner 标记为 closed。

closed 后继续调用 initialize、advance 或 reset 都会报错。

## 8. `ops/cuda/beam_state.py`：设备侧状态机

### 8.1 `BeamStateScratch`

它是每一步的临时区，不承载跨步业务状态。主要字段包括：

- `parent_scores`
- `selection_mask`
- `terminal_ranks`
- `finish_reasons`
- `cumulative_scores`
- `row_counts`
- `row_errors`
- `row_valid_counts`
- `row_offsets`
- `execute`
- `reorder`
- `next_completed`

其中 `execute` 和 `reorder` 是关键设备 latch：即使固定 graph 中后续 kernel 仍被 launch，也可以通过 latch 阻止它们真正修改 State/KV。

### 8.2 `_prepare_candidates_kernel()`

这个 kernel 对候选进行打分、合法性检查和 active/completed 分类。

Decode 候选累计分数为：

```text
cumulative_score[parent, rank]
    = beam_scores[parent] + logprobs[parent, rank]
```

Prefill 没有已有 Beam 分数，root score 视为 0。

候选有效需要同时满足：

- `valid_mask=True`；
- parent 处于 active 状态；
- token ID 非负；
- local logprob 和 cumulative score 都不是 NaN/Inf。

随后判断终止原因：

```text
命中 stop token → FINISH_STOP
达到 max token → FINISH_LENGTH
否则           → 可参与下一轮 active selection
```

当 stop 和 length 同时成立时，stop 优先。

kernel 还输出每个 parent 的：

- completed 数量；
- valid 数量；
- candidate error 数量。

### 8.3 `_validate_step_kernel()`

这个 kernel 在修改持久状态和 KV 之前统一决定本轮能否执行。

主要错误包括：

- Decode 发生在 Prefill 初始化前；
- Prefill 重复初始化；
- candidate token/score 非法；
- completed storage 容量不足；
- generation/decode counters 不一致；
- Decode KV 位置超过 KV capacity；
- 没有任何有效候选，且之前也没有 completed result。

错误写入 `error_code` 后保持不变，直到 reset。错误发生时：

```text
execute = False
reorder = False
```

因此后续固定 kernel 可以被 enqueue，但不会部分修改持久 Beam State 或 KV。

已经 finished 的 Session 也会令 `execute=False`。这样终态后的额外 graph replay 是状态 no-op。

### 8.4 `_append_completed_kernel()`

把本轮所有 terminal candidates 追加到 `completed_*`。

每条记录保存：

- 最后一个 token；
- 原始 cumulative score；
- parent ID；
- 完成长度；
- finish reason；
- flat candidate tie-break index。

记录顺序按 generation step、parent、candidate rank 确定，并通过 per-row prefix sum 计算写入位置，没有使用无序 atomic reservation。

### 8.5 `_stage_sequence_kernel()`

根据每个 selected child 的 parent ID，从旧 sequence 复制有效前缀到 Workspace 的 `next_sequence`，然后追加新 token。

它先写 Workspace 而不是直接原地写 State，是为了处理 duplicate parent 和交叉 parent mapping，避免输入 sequence 在被其他 child 读取前遭到覆盖。

### 8.6 `_commit_beams_kernel()`

将选择结果正式写回：

- `sequence`
- `beam_scores`
- `input_ids`
- `parent_ids`
- `active_mask`
- `sequence_lengths`
- `history_token_ids`
- `history_parent_ids`

不足 W 个 active Beam 时仍保留固定 W 行。无效行使用：

```text
active_mask = False
score       = -inf
token       = 0
parent      = 0
length      = 0
```

后续模型和 KV kernel 必须通过 mask 忽略这些物理占位行。

### 8.7 `_commit_step_kernel()`

这是当前 step 的最终提交点，更新：

- `num_generated_tokens`
- `num_decode_steps`
- `num_active_beams`
- `num_completed`
- `is_initialized`
- `is_finished`

它位于 KV reorder 之后。因此计数器前进意味着 active Beam state 和相应 KV mapping 已经一起完成。

## 9. `beam_selection.py`：复用已有 Top-W

A4 没有实现另一套 Beam Top-K，而是复用 `_hierarchical_topk_out()` 和 `_group_beam_outputs_kernel()`。

### 9.1 `_hierarchical_topk_out()`

从所有非终态候选中选择下一轮 Top-W。

确定性排序规则是：

1. cumulative score 降序；
2. 分数相同时，flattened candidate index 升序。

A4 新增 `mask_invalid_indices=True`。这样无效 candidate 不仅 score 为 `-inf`，其 flat index 也会被传播为无效值，后续可以准确区分：

- 真正分数为 `-inf` 的合法候选；
- 纯物理占位的无效候选。

### 9.2 `_group_beam_outputs_kernel()`

它把选中的 flat index 转换为：

```text
parent_id      = flat_index // W
candidate_rank = flat_index % W
```

并写出 selected token、parent、score 和 flat index。

## 10. `beam_kv_reorder.py`：按 parent 重排 KV

KV 重排分成两个 kernel。

### 10.1 `_select_kv_gather_kernel()`

按照 selected `parent_ids`，把所有层、所有已写 Decode position 的 parent KV 复制到独立 aux buffer。

### 10.2 `_select_kv_scatter_kernel()`

再把 aux buffer 写入新的 child Beam row。

必须使用 gather → scatter，不能直接原地 copy。例如：

```text
child 0 ← parent 1
child 1 ← parent 1
child 2 ← parent 0
```

直接原地更新 child 0 后，可能破坏 child 2 仍需读取的 old row 0；aux buffer 则保留了完整旧状态。

KV kernel 使用 GPU 上的：

- `num_decode_steps`
- `active_mask`
- `execute/reorder`

因此不需要 CPU 读取当前 step 或 active 数量。

## 11. 对现有模块的兼容性修改

### 11.1 `beam_decision.py`

`BeamPostprocessWorkspace` 新增可选的 `state_scratch`。

- 只有绑定 A4 Runner 时才分配；
- 普通旧路径维持原来的分配和接口；
- `device_tensors()` 和显存统计包含新 scratch。

### 11.2 `beam_attention_context.py`

`BeamDecisionBuffers.reset()` 新增可选 `batch_index`：

```python
decision.reset()            # 兼容旧行为，重置全部 row
decision.reset(state_index) # A4，仅重置指定 row
```

Context 还预先建立所有层的 KV pointer tables。Runner 可直接复用固定地址，避免每一步动态创建 GPU 指针 Tensor。

### 11.3 `beam_search_session.py`

`begin_session()` 新增 `decision_index`，允许只清理当前 State row，而不是清空整个 decision buffer。

`end_session()` 保持原职责：结束 Session 并把 KV slot 还给 registry。跨 stream 等待由 Runner 外层处理。

### 11.4 `beam_kernels.py`

原来的 KV gather/scatter kernel 被抽到公共 `ops/cuda/beam_kv_reorder.py`。

旧 `select_unshared_kv_fused()` 与新 A4 Runner 复用同一份 kernel，避免两套 KV reorder 逻辑逐渐产生差异。

## 12. 完整执行流程

下面从一次请求开始，串起所有主要函数。

### 12.1 创建和绑定资源

预期的上层代码首先取得已有 Context、Session 和 Workspace，然后创建 State/Runner：

```python
session.begin_session(
    beam_width=W,
    prefix_len=prompt_len,
    decision_index=state_index,
)

state = BeamSearchState(
    context.decision,
    beam_width=W,
    max_new_tokens=T,
    stop_token_ids=stop_ids,
    ignore_stop_tokens=False,
)

runner = BeamSearchRunner(
    session,
    state,
    workspace,
    state_index=state_index,
)
```

完成后，State、Workspace 和 KV 的物理地址固定。

### 12.2 Prefill 初始化

模型完成 Prefill，Constraint/Sampling 在 GPU 上产生 `[1,W]` candidates：

```python
runner.initialize_from_prefill(token_ids, logprobs, valid_mask)
```

内部调用顺序：

```text
_prepare_candidates_kernel
        ↓
以 root score=0 计算候选累计分数，区分 active/completed
        ↓
_validate_step_kernel
        ↓
确认没有重复初始化、候选合法且容量足够
        ↓
_hierarchical_topk_out
        ↓
选择最多 W 个 active candidates
        ↓
_group_beam_outputs_kernel
        ↓
得到 selected token/parent/score
        ↓
_append_completed_kernel
        ↓
保存 Prefill 阶段已经完成的候选
        ↓
_stage_sequence_kernel
        ↓
构造长度为 1 的 next sequence
        ↓
_commit_beams_kernel
        ↓
写入 active Beam、下一轮 input 和 history
        ↓
不做 KV reorder
        ↓
_commit_step_kernel
```

成功后：

```text
num_generated_tokens = 1
num_decode_steps      = 0
is_initialized        = True
```

`state.input_ids` 可以直接作为第一次 Decode 的输入。

### 12.3 一次 Decode forward

ModelRunner 从 State 取得：

- `input_ids`
- `active_mask`
- `sequence_lengths`
- 对应 KV Pool slot

执行模型 forward，并把新 KV 写到：

```text
KV pool[..., num_decode_steps, :]
```

第一次 Decode 时 `num_decode_steps=0`，因此写 Decode KV position 0。

Constraint/Sampling 随后生成 `[W,W]` candidates。

### 12.4 推进一次 Decode Beam

调用：

```python
runner.advance(token_ids, logprobs, valid_mask)
```

内部调用链：

```text
_prepare_candidates_kernel
        ↓
local logprob + parent beam score
        ↓
EOS/length → completed
其他合法项 → active selection
        ↓
_validate_step_kernel
        ↓
统一检查状态、计数、容量、KV position 和候选
        ↓
_hierarchical_topk_out
        ↓
从最多 W² 个 active candidates 中选择 Top-W
        ↓
_group_beam_outputs_kernel
        ↓
flat index 转成 token/parent/score
        ↓
_append_completed_kernel
        ↓
保存所有 terminal candidates
        ↓
_stage_sequence_kernel
        ↓
按 selected parent 构造 next sequence
        ↓
_commit_beams_kernel
        ↓
提交 active Beam 和 append-only history
        ↓
_select_kv_gather_kernel
        ↓
按 parent 把所有旧 KV 收集到 aux
        ↓
_select_kv_scatter_kernel
        ↓
把 aux 写入新的 Beam row
        ↓
_commit_step_kernel
```

成功后：

```text
num_generated_tokens += 1
num_decode_steps      += 1
```

下一轮 forward 看到的 token、sequence、score、parent 和 KV 已经一致。

### 12.5 EOS 和长度终止

部分候选结束时：

- completed candidates 追加到 `completed_*`；
- 其他候选继续竞争 active Top-W。

所有候选结束或达到最后一轮时：

- completed records 保留；
- `active_mask` 全部变为 false；
- `is_finished=True`；
- 不再执行 KV copy，因为不存在下一轮 active state。

如果固定 CUDA Graph 在 finished 后又 replay：

- `_validate_step_kernel()` 设置 `execute=False`；
- 后面的 State/KV kernel 成为 no-op；
- 已保存的 completion 和 KV 不被破坏。

但 A4 #383 不能阻止 graph 外面的模型 forward。终态后不再调模型，是后续 ModelRunner 集成层的责任。

### 12.6 A5 如何重建最终序列

A4 保存的是完成记录和 append-only history，不直接产生用户输出。

对长度为 n 的 completed record：

1. 从该 record 的 `parent_id` 开始；
2. 读取 history 的第 `n-2` 层 token；
3. 沿该层的 parent 指向上一层；
4. 一直回溯到隐式 root；
5. 将回溯 token 反转；
6. 追加 completed record 的最后一个 token。

A5 还需要执行：

- length penalty；
- 最终 Top-R；
- stop token 保留/删除；
- 紧凑结果构造和必要的最终 D2H。

这些都不属于 A4 #383。

### 12.7 reset 和资源释放

如果资源将用于同配置的新请求：

```text
等待所有最终消费者
        ↓
runner.reset()
        ↓
session.begin_session(..., decision_index=同一 row)
        ↓
initialize_from_prefill()
```

如果 Session 生命周期结束：

```python
runner.end_session(consumer_done)
```

它在确认生产者和最终消费者都完成后，才把 KV slot 归还 registry。

## 13. 错误、并发和 CUDA Graph 语义

### 13.1 sticky error

设备错误一旦写入 `error_code`，后续调用不会继续修改 State/KV，直到 reset。

这样可以防止：

- sequence 已更新但 KV 未更新；
- completed count 已增加但 completed record 未完整写入；
- counter 前进但当前 step 实际失败。

它不是事务回滚机制。CUDA 执行本身发生不可恢复错误时，仍然需要上层终止 Session。

### 13.2 当前并发限制

虽然 State 有 `Bmax` 维度，但 PR #383 当前只允许一个 active runner 使用共享 Workspace/Context。

真正并发执行多个 Session 还需要：

- execution → state row/KV slot 的明确映射；
- 分区或独立的 KV aux buffer；
- 分区 Workspace；
- 对共享 CUDA Graph 和 stream 的并发规则。

所以目前的 batch dimension 更像存储层的扩展准备，不代表已经支持并发 Beam execution。

### 13.3 stream 所有权

`initialize_from_prefill()` 和 `advance()` 都在调用者当前 CUDA stream enqueue。

调用者必须保证：

- 如果 candidates 来自另一条 stream，本 stream 先等待 producer event；
- 如果最终消费者在另一条 stream，reset/release 前等待 consumer event；
- graph 释放前停止所有使用该资源绑定的 replay。

### 13.4 CUDA Graph 固定项和动态项

Capture 后固定：

- State row；
- KV slot；
- Workspace；
- candidate Tensor 地址；
- Beam width 和容量；
- kernel launch grid。

Replay 时动态读取：

- generated/decode counters；
- active mask；
- candidate values；
- valid mask；
- finished/error 状态。

因此同一个 graph 可以处理 active Beam 数从多到少、EOS 终止以及终态后的 no-op，但不能在不 recapture 的情况下换掉资源地址或固定配置。

## 14. A4 与 A3 的关系

A3 和 A4 分别负责控制面与数据面。

### 14.1 A3 已经提供什么

A3 在 EngineCore/Scheduler 中维护：

- Session 生命周期；
- issued/completed stage；
- 两阶段 look-ahead；
- Prefill/Decode stage 编号；
- `GRDeviceStateRef(session_id, stage_index)`；
- `GRStageMetadata`；
- abort、failure、drain 和 terminal result 通道。

A3 的目标是：CPU 可以在未消费上一阶段 Beam payload 的情况下，把下一 stage 放进 native async queue。

### 14.2 A4 #383 已经提供什么

A4 接收已经位于 GPU 的 candidates，并维护：

- active Beam token/score/parent；
- completed records；
- history；
- GPU counters；
- KV parent reorder；
- fixed-address graph replay；
- stream-aware reset/release。

### 14.3 两者目前缺少的连接

当前上游还没有代码把 A3 的 stage metadata 映射成 A4 Runner 调用。后续 A4 PR2 大致需要实现：

```text
A3 Scheduler 产生 GRStageMetadata
        ↓
ModelRunner 按 session_id 找到 BeamSearchRunner
        ↓
等待 input_device_state 对应的 CUDA event
        ↓
从 BeamSearchState 准备本轮 input token / position / KV
        ↓
执行模型 forward
        ↓
Constraint/Sampling 生成 GPU candidates
        ↓
stage 0: initialize_from_prefill()
stage k: advance()
        ↓
记录 output_device_state 的完成 event
        ↓
返回轻量 GRWorkerResult 给 A3
```

这层集成必须特别保证：

1. 模型写 KV 先于 `advance()`。
2. `advance()` 完成先于下一 stage 消费 State/KV。
3. 同一 parent mapping 只重排一次 KV。
4. finished/error 后不再执行模型 forward。
5. A5 最终读取完成前不 reset State。
6. 所有 consumer 完成前不释放 KV slot。
7. A3 已排队但需要 drain 的 successor 不会访问已释放资源。

## 15. 阅读时最值得关注的问题

读完主流程后，可以重点检查以下几点：

1. `num_generated_tokens` 和 `num_decode_steps` 的语义不同。Prefill 后前者为 1、后者为 0。
2. 本轮模型先写 KV，A4 再按同一步产生的 parent mapping 重排，最后才提交 counter。
3. completed candidates 和 active Top-W 是两条独立数据流，EOS 不能只看 active Top-W。
4. history 是 append-only；active sequence 和 KV 可以反复覆盖。
5. `execute/reorder` latch 保证错误或终态 replay 不部分更新持久状态。
6. KV slot、state index 和 workspace row 是三个不同概念。
7. `end_session()` 的同步位于生命周期边界，不属于 steady Decode path。
8. completed capacity 按 W²T 增长，正确性保守但显存成本较高。
9. A4 PR1 不读取 A3 metadata，也没有生产入口；性能收益要等 PR2 接入后才能做端到端判断。

## 16. 最简心智模型

如果只记住一张图，可以记成：

```text
                    ┌──────────────────────────┐
model / constraint │ token + logprob + valid  │
                    └────────────┬─────────────┘
                                 ↓
                    classify active/completed
                         ↓                 ↓
                  active candidates   completed records
                         ↓                 ↓
                  deterministic Top-W  append-only history
                         ↓
                  token/parent/score
                         ↓
              update sequence + reorder KV
                         ↓
                  commit device counters
                         ↓
                  next model Decode step
```

A4 的核心不是单独某个 Top-K kernel，而是让候选分类、Beam 状态、完成历史、KV mapping 和计数器在同一条设备执行链中保持一致，并把 CPU 从每个 Decode step 的关键依赖路径中移走。
