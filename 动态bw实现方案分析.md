# 基于 xAttention 的动态 Beam Width 适配分析

## 1. 固定 BW 与动态 BW 的差异

### 1.1 当前固定 BW

当前实现中，一个 Beam Search 请求从开始到结束使用同一个 `beam_width=W`：

```text
Prefill → W → W → W → 输出
```

整个流程都可以基于固定的 `W` 构建：

```text
Decode 输入       [B × W]
Unshared KV       [B, W, KVHeads, MaxSteps, HeadDim]
Beam scores       [B, W]
Sequence          [B, W, MaxSteps]
TopK 输出         [B, W]
```

`BeamSearchContext`、xAttention、Beam TopK、KV 重排和 ACL Graph 都默认本轮搜索中的 `W` 不变。

### 1.2 动态 BW

动态 BW 是指不同 decode step 使用不同的活跃 beam 数，例如：

```text
Prefill → W1=128 → W2=64 → W3=32 → 输出
```

每次 BW 变化后，下一步参与模型计算的 token 数会发生变化：

```text
B × W_current → B × W_next
```

与固定 BW 相比，动态 BW 会带来以下变化：

| 模块 | 固定 BW | 动态 BW |
|---|---|---|
| Decode token 数 | 始终为 `B×W` | 每个 step 可能不同 |
| Beam 状态 | 固定 `[B,W]` | 需要按当前 BW 更新有效区域 |
| Unshared KV | 固定 beam 维 | 需要收缩、扩展或重新映射 |
| Beam TopK | 保留固定 W 个结果 | 从当前候选中保留下一步 `W_next` 个结果 |
| Parent index | 固定范围 `[0,W)` | 输入范围和输出数量随 step 变化 |
| ACL Graph | 一套固定 shape | 需要多个 shape/bucket |
| Scheduler | 只关心固定 `B×W` | 需要感知当前 step 的实际 BW |

因此，动态 BW 不是简单修改一次 `num_beams`，而是需要让整个 Beam Search 数据流都显式区分当前宽度和下一步宽度。

## 2. 架构上的主要调整

### 2.1 引入 Beam Width Schedule

当前只有一个全局 `beam_width`。动态 BW 需要增加一个宽度计划，例如：

```python
beam_width_schedule = [128, 64, 32]
```

Beam Search Context 在每一步至少需要记录：

```text
current_width       当前参与模型计算的 beam 数
next_width          Beam Select 后保留的 beam 数
max_beam_width      BufferPool 的最大容量
candidate_top_k     每个当前 beam 产生的候选数
```

其中 `current_width`、`candidate_top_k` 和 `next_width` 不能再默认相等。

### 2.2 Beam Select 语义调整

固定 BW 的选择逻辑通常是：

```text
W 个 parent
× 每个 parent 的 Top-W 候选
→ 全局保留 W 个 beam
```

动态 BW 应改成：

```text
W_current 个 parent
× 每个 parent 的 candidate_top_k 个候选
→ 全局保留 W_next 个 beam
```

即 Beam Select 的输入输出 shape 分别由不同参数决定：

```text
输入 beam scores     [B, W_current]
候选 token/prob     [B×W_current, candidate_top_k]
输出 token/index    [B, W_next]
```

这是动态 BW 最核心的架构变化。若 Beam TopK 算子仍然只接收一个 `top_k`，就无法准确表达动态 BW。

### 2.3 Beam 状态使用“容量 + 有效宽度”

不建议每次 BW 变化都重新分配 Beam 状态和 KV。更合适的方式是：

```text
物理容量：max_beam_width
有效区域：current_width
```

例如 Pool 按 128 分配，而当前 step 只使用前 32 个 beam：

```text
Beam scores    [B, 128]，有效部分为 [:, :32]
Sequence       [B, 128, T]，有效部分为 [:, :32, :]
Unshared KV    [B, 128, ...]，有效部分为 [:, :32, ...]
```

这样可以保持 BufferPool 地址稳定，降低重新分配对 ACL Graph 的影响。

需要注意：后续逻辑必须使用 `current_width`，不能继续从张量的物理 shape 推断真实 BW。

### 2.4 Unshared KV 的重新映射

每次 Beam Select 都会产生：

```text
new_beam → parent_beam
```

动态 BW 下存在两种情况。

#### BW 收缩

```text
W_current=128 → W_next=32
```

只需要保留被选中的 32 个 parent 路径，并将对应历史 KV 整理到下一步使用的有效区域。

#### BW 扩展

```text
W_current=32 → W_next=64
```

多个新 beam 可能来自同一个 parent，需要复制同一份历史 KV，再在后续 step 分别追加新的 KV。

当前 `select_unshared_kv` 使用物理 KV 重排。它可以作为第一版实现，但 BW 频繁变化时，重排开销大致随以下规模增长：

```text
Layers × Batch × W_next × DecodedSteps × KVSize
```

长期更合理的方向是保存 parent index/BeamPath，让 xAttention 根据逻辑父子关系读取历史 KV，减少历史 KV 的反复复制。

### 2.5 xAttention 增加有效宽度信息

xAttention 当前可以从输入 query 数量和 KV shape 间接推断 BW，但采用最大容量 BufferPool 后：

```text
KV.shape[1] = max_beam_width
实际有效 BW = current_width
```

因此 xAttention 或其 metadata 需要显式携带：

```text
current_width
active_token_count
request/beam mapping
```

算子只应处理有效 beam，不能读取 Pool 中未使用的 beam slot。

如果使用 BW bucket 和 padding，还需要 `active_beam_mask` 来屏蔽填充 beam。

### 2.6 ACL Graph 改为多 BW Bucket

当前 Graph runtime shape 与 `B×W` 相关。动态 BW 会使每个 step 的 shape 发生变化。

比较可行的方式是预先支持有限的 BW bucket：

```text
8、16、32、64、128
```

实际 BW 向上匹配最近 bucket：

```text
W=48 → bucket=64
```

Graph cache key 至少需要区分：

```text
(batch_bucket, beam_bucket, decode_step_bucket)
```

不能只用 `B×W` 作为 key，因为：

```text
B=1, W=128
B=2, W=64
```

虽然 token 总数相同，但 request 分组、Shared KV block table 和 Beam TopK 分组完全不同。

### 2.7 Scheduler 与多请求策略

第一版建议限制：

- 同一 batch 内所有请求使用相同的 BW schedule；
- 当前 step 的 BW 相同；
- 不同 BW 或不同 schedule 的请求进入不同 bucket。

这样仍可使用规则的 `[B,W,...]` 布局。

如果同一 batch 内不同请求使用不同 BW，就需要使用 ragged layout：

```text
beam_widths          [B]
beam_offsets         [B+1]
总 token 数          sum(beam_widths)
```

这会显著增加 Scheduler、xAttention、KV mapping 和 Beam TopK 的复杂度，不建议作为第一版目标。

## 3. 主要影响

### 3.1 性能

动态 BW 的主要收益是逐步减少参与后续 forward 的 beam 数。例如从 128 收缩到 32 后，后续模型计算量和当前 step KV 写入量会明显下降。

新增开销主要包括：

- BW 切换时的 Beam 状态整理；
- Unshared KV 的收缩、扩展和物理重排；
- 多套 ACL Graph 的管理；
- bucket padding 带来的无效计算。

是否有收益主要取决于：

```text
后续 forward 节省的计算
是否大于
BW 切换和 KV 重排的额外成本
```

动态 BW 更适合前期宽、后期快速收缩的短序列推荐场景。

### 3.2 显存

如果 BufferPool 仍按最大 BW 分配，动态 BW 不会明显减少常驻显存，只会减少有效计算和访问范围。

如果希望同时降低显存，需要进一步实现：

- 按 BW bucket 建立不同规格的 Pool；
- Pool 的回收或 LRU 管理；
- BeamPath/逻辑 KV slot，避免按最大 BW 长期保留全部空间。

### 3.3 精度和一致性

动态 BW 本身会改变搜索空间，因此输出不应要求与固定最大 BW 完全一致。

需要重点保证的是：

- 每次收缩都从全部合法候选中选择 `W_next`；
- cumulative beam score 正确继承；
- parent index 属于当前 request，不能跨 request；
- KV、sequence 和 score 使用同一个 parent mapping；
- padding beam 不参与 TopK；
- eager 与 ACL Graph 结果一致。

## 4. 推荐实现方案

建议第一版采用：

```text
固定 BW schedule
+ 同 batch 同宽
+ 最大容量 BufferPool
+ 常用 BW Graph bucket
+ select_unshared_kv 物理重排
```

示例：

```text
beam_width_schedule = [128, 64, 32]
candidate_top_k      = [32, 16, 8]
```

执行流程：

```text
Step N 使用 W_current 做模型 forward
    ↓
每个 beam 生成 candidate_top_k 个候选
    ↓
全局选择 W_next 个 beam
    ↓
按 parent index 更新 score、sequence 和 Unshared KV
    ↓
下一步以 B×W_next 个 token 执行
```

在该版本稳定后，再考虑：

- 任意 BW 和 padding mask；
- 同 batch 不同 BW；
- 使用 BeamPath 替代物理 KV 重排。

## 5. 代码修改点与难度

| 修改位置 | 主要修改 | 难度 |
|---|---|---|
| `vllm/vllm/entrypoints/llm.py` | 接收 BW schedule；每步传递 current/next width；按动态输出更新 CPU Beam 状态 | 中 |
| `vllm_ascend/beam_search/context.py` | 保存 current/next/max width；动态切换有效 slice；按 `W_next` 更新状态 | 中高 |
| `vllm_ascend/ops/xllm_ops.py` | 拆分 `candidate_top_k` 与 `next_width`；向算子传递有效 BW/mask | 中 |
| xllm 自定义算子 | 修改 Beam TopK 输出 shape；支持动态 parent 数、有效 BW 和重复 parent | 高 |
| `vllm_ascend/attention/attention_v1.py` | 向 xAttention 传递当前有效 BW；保证 query、KV 和 metadata 一致 | 中 |
| `vllm_ascend/worker/model_runner_v1.py` | 动态构造 input IDs、positions、logits indices 和 token 数；切换 Graph bucket | 高 |
| `vllm_ascend/compilation/acl_graph.py` | Graph 按 batch/beam bucket 管理；更新动态 metadata | 高 |
| Scheduler/batching | 将不同 BW schedule 的请求分桶；若支持同批不同 BW 则引入 ragged mapping | 中高/高 |

整体来看：

- 只支持几个固定 schedule、同 batch 同宽：**中高难度**；
- 支持任意 BW 和自动 bucket：**高难度**；
- 支持同 batch 不同 BW：**很高难度**。

最大的不确定性在 xllm 自定义算子。当前仓库主要是 Python wrapper，必须确认底层 `x_attention`、`beam_search_group` 和 `select_unshared_kv` 对动态 shape、最大 TopK 和 Graph replay 的实际支持能力。

## 6. 总结

动态 BW 的核心不是让一个 `num_beams` 参数随 step 变化，而是将固定 BW 架构调整为：

```text
最大物理容量
+ 当前有效宽度
+ 下一步目标宽度
+ 动态 Beam parent mapping
+ 多 BW Graph bucket
```

推荐先实现“**固定 schedule、同 batch 同宽、有限 BW bucket**”。该方案能获得后期减少 beam forward 的主要收益，同时避免一开始就引入 ragged batch 和完整的动态 KV 管理。
