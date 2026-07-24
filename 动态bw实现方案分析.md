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



# 基于 Tree Attention 的动态 Beam Width 设计

## 1. 目标与范围

本文讨论在现有 Tree Attention beam search 上支持“不同生成层采用不同候选保留数”的工程方案。设第 \(d\) 层裁剪后保留的活跃 beam 数为 \(B_d\)，例如：

```text
depth       0    1    2    3    4+
beam width  4   16   64   32    8
```

这里的“层”指生成树的深度，即生成第几个 token，而不是 Transformer 的网络层。本文关注搜索过程、调度器、KV cache 和 Tree Attention 元数据的变化，不讨论动态宽度策略本身如何产生。

## 2. 现有 Tree Attention 基础

固定宽度 beam search 每一步包含以下过程：

1. 当前活跃 beam 分别计算下一个 token 的候选及分数；
2. 将所有父 beam 的候选展平并做全局 Top-K；
3. 记录每个入选候选的父 beam，以及该候选 token；
4. 下一轮通过 parent/child 映射 fork 逻辑分支；
5. Tree Attention 复用公共前缀 KV，只为各分支维护新增的后缀 KV。

当前实现中的关键抽象是 `fork_info`（父 beam 下标与新 token 的对应关系）以及每个请求携带的活跃 `beam_width`。因此，Tree Attention 本身已经能表达一棵非完整树：一个父节点可没有子节点、只有一个子节点或产生多个子节点。动态宽度的核心不是重写 attention，而是让搜索层和调度层不再假设每一层的活跃分支数恒定。

## 3. 总体方案

### 3.1 配置模型

请求增加按深度索引的宽度序列：

```text
beam_width_schedule = [B0, B1, ..., Bm]
```

生成深度超过序列长度时，需要明确一种规则：

- 重复最后一个值；
- 回退到默认宽度；
- 将 schedule 视为非法。

推荐重复最后一个值，行为稳定且适合长度未知的生成请求。所有值必须为正整数，并设置服务端上限，避免单层宽度导致显存或候选排序开销失控。

最终返回结果数应与“中间活跃宽度”分开定义。建议保留独立的 `num_return_sequences`；否则应明确最终返回数由最后一层宽度决定。将两者混为同一个 `beam_width`，会在中间宽度小于最终返回数时产生语义冲突。

### 3.2 每层搜索流程

在深度 \(d\)，令上一层活跃 beam 数为 \(B_{d-1}\)，每个父 beam 取 \(K_d\) 个 token 候选。搜索层执行：

```text
parents(B[d-1])
    -> logits / per-parent Top-K(K[d])
    -> flatten candidates(B[d-1] * K[d])
    -> global Top-K(B[d])
    -> parent index + token
    -> fork_info for next tree-attention step
```

全局裁剪的目标值由固定 `beam_width` 改成当前层的 \(B_d\)。入选项在展平数组中的位置为 \(i\) 时：

```text
parent_index = i // K_d
token_index  = i % K_d
```

不能再使用固定宽度充当候选 stride。若 \(K_d\) 也动态变化，stride 必须作为该步元数据显式传递，否则父节点映射会错误，继而 fork 到错误的 KV 分支。

### 3.3 与 Tree Attention 的衔接

裁剪完成后，只为入选的 \(B_d\) 个候选生成下一步 `fork_info`。下一轮 Tree Attention 读取实际活跃宽度，而不是请求初始宽度：

```text
active_beam_width = len(fork_info) = B[d]
```

宽度扩大时，多个新子节点可以引用同一父节点；底层只共享父节点已有 KV，并为各子节点追加后缀。宽度缩小时，未入选分支应从活跃集合移除，其 KV block 引用计数递减；确认不再被活跃分支或已完成序列引用后才能释放。

所以 attention 数学计算本身不需要变化，变化主要集中在：

- 当前步的活跃序列数量；
- query、slot mapping 和 block table 的长度；
- 父子分支映射；
- 每个逻辑请求在 batch 中的分段边界；
- 被淘汰分支的 KV 生命周期。

## 4. 需要修改的模块

### 4.1 API 与参数校验

增加宽度 schedule、最终返回数以及可选的每层候选数 \(K_d\)。校验内容包括：

- schedule 非空且元素为正整数；
- 最大宽度不超过服务端限制；
- 最终返回数不超过可能产生的有效完成序列数；
- graph capture、最大 batch token 和 KV cache 容量能覆盖最大宽度；
- 在线接口、离线接口和序列化协议语义一致。

### 4.2 Beam 搜索与候选排序

搜索循环需按深度读取 \(B_d\)，并使用它进行全局 Top-K。完成序列和活跃序列必须分开管理：EOS 分支进入 completed 集合后不应占用下一层活跃宽度，但最终排序时仍参与竞争。

若宽度从小变大，扩展能力还受每个父节点输出候选数 \(K_d\) 限制，必须满足：

\[
B_d \le B_{d-1}K_d
\]

否则期望宽度无法达到。Catalog、语法约束或大量 EOS 还可能使有效候选数进一步下降，因此运行时实际宽度应允许小于配置宽度。

### 4.3 调度器与批处理元数据

固定宽度实现容易预先按 `batch_size × beam_width` 分配和切片。动态宽度后，每个请求的宽度、深度和完成状态都可能不同，需要使用 prefix-sum/offset 描述 ragged batch：

```text
request offsets = [0, B0, B0+B1, ...]
```

请求在每一步都要更新：

- 活跃 beam 数；
- logits 输出分段；
- fork 的父下标；
- request-to-sequence 映射；
- query length、slot mapping 和 block table offset。

这是实现中最容易出现跨请求数据串位的部分。不能用 batch 内最大宽度简单推导每个请求的真实边界，除非明确 padding 并在排序及 fork 时屏蔽 padding 项。

### 4.4 KV Cache 管理

动态宽度不改变公共前缀共享原则，但增加了引用计数变化的频率：

- 扩宽：同一父 block table 被多个子分支引用；
- 缩宽：一次释放多个淘汰分支；
- EOS：分支从活跃集合转入完成集合；
- early stop：整个请求的剩余分支统一回收。

若完成序列只保留 token、分数和 parent pointer，通常无需继续持有其完整 KV；若后续存在回溯或重新扩展，则必须延长 KV 生命周期。实现时要明确 completed beam 是否仍拥有 KV。

### 4.5 Kernel 与 CUDA Graph

若现有 Tree Attention kernel 接受每请求宽度数组和 offset，通常无需修改核心 kernel，只需传递真实 shape。若 kernel 将 beam width 编译为常量，或 CUDA Graph 只捕获固定形状，则需要：

- 为若干宽度 bucket 预捕获 graph；
- 对宽度做 padding 并提供有效 mask；
- 或对不匹配宽度回退 eager execution。

完全动态 shape 会减少 graph 复用率；全量 padding 则浪费算力。工程上通常采用宽度 bucket，例如 8/16/32/64，并将 \(B_d\) 向上取整到最近 bucket。

## 5. 实现难度

整体难度为中高。若现有 Tree Attention 已显式携带 parent/child 映射和实际活跃宽度，搜索层改造属于中等难度；真正复杂的是 batch、KV 生命周期和图捕获。

| 部分 | 难度 | 原因 |
|---|---:|---|
| 参数与按层 Top-K | 低至中 | 将固定 K 改为按深度读取，逻辑直接 |
| parent/child 映射 | 中 | 动态候选 stride 和重复父节点必须正确 |
| Ragged batch 元数据 | 高 | 不同请求宽度不同，offset 易错 |
| KV cache fork/free | 高 | 扩宽、缩宽、EOS 并发改变引用关系 |
| CUDA Graph/静态 shape | 高 | 动态宽度降低图复用，需 bucket 或 fallback |
| 分布式 TP/DP 同步 | 中至高 | 各 rank 必须得到一致的 Top-K 和分支顺序 |
| 正确性验证 | 高 | 组合状态多，错误常表现为静默生成偏差 |

相比从普通 attention 新建 Tree Attention，动态宽度不需要重新解决公共前缀共享，因此难度明显较低；但它会打破固定宽度在内存布局、batch 切片和 graph capture 上带来的大量简化。

## 6. 复杂度分析

设生成深度为 \(D\)，第 \(d\) 层裁剪后的活跃宽度为 \(B_d\)，每个父 beam 的候选数为 \(K_d\)，词表大小为 \(V\)，当前上下文长度为 \(L_d\)。

### 6.1 候选生成和裁剪

若 logits 已计算，只分析候选选择：

- 每父节点词表 Top-K：约为 \(O(B_{d-1}V)\)，具体常数取决于 kernel；
- 候选合并数量：\(C_d=B_{d-1}K_d\)；
- 使用 partial Top-K/selection：期望 \(O(C_d)\) 或 kernel 相关的 \(O(C_d\log B_d)\)；
- 完整排序：\(O(C_d\log C_d)\)，不推荐。

总候选处理量为：

\[
O\left(\sum_{d=1}^{D} B_{d-1}K_d\right)
\]

固定宽度 \(B\)、固定候选数 \(K\) 时退化为 \(O(DBK)\)。动态宽度只有在多数层的 \(B_d\) 小于固定基线时才降低总量；单层扩到很大仍会造成明显峰值。

### 6.2 Tree Attention 计算

在共享公共前缀 KV 的前提下，每层 query 数与活跃 beam 数成正比。抽象表示为：

\[
T_{\text{attn}}
=
O\left(\sum_{d=1}^{D} B_{d-1}L_dH\right)
\]

其中 \(H\) 表示与 head 数、head dimension 有关的因子。相较固定宽度：

\[
O\left(B\sum_{d=1}^{D}L_dH\right)
\]

动态宽度节省比例主要由 \(\sum B_d\) 决定，而不是最大宽度决定。但 kernel padding 或宽度 bucket 会使实际计算量接近 \(\sum \widehat{B_d}\)，其中 \(\widehat{B_d}\) 是向上取整后的物理宽度。

### 6.3 KV Cache 空间

公共 prompt KV 只保存一份。若每层每个活跃分支追加一个 token，理想化的增量 KV 空间为：

\[
S_{\text{KV}}
=
O\left(S_{\text{prompt}}+\sum_{d=1}^{D}B_d\right)
\]

实际峰值还取决于 block 粒度、copy-on-write、延迟释放和 completed beam 是否持有 KV。动态宽度的平均空间可能下降，但最大瞬时空间至少要按：

\[
O\left(S_{\text{prompt}}+D\cdot \max_d B_d\right)
\]

做保守容量规划。若某层突然扩宽，可能出现 KV block 分配尖峰。

### 6.4 元数据与通信

每步 parent/child、slot mapping、block table 等元数据规模为 \(O(B_d)\)，总量为：

\[
O\left(\sum_{d=1}^{D}B_d\right)
\]

在张量并行中，候选分数归并和 Top-K 通信还与 \(B_{d-1}K_d\) 有关。为了保证各 rank 的分支顺序一致，分数相同场景必须定义稳定的 tie-break 规则。

## 7. 主要风险

1. **父下标计算错误**：仍使用固定候选 stride，会 fork 到错误分支。
2. **EOS 占用活跃配额**：completed 与 active 未分离，导致下一层宽度小于预期。
3. **宽度扩大但候选不足**：\(B_d>B_{d-1}K_d\)，配置目标不可达。
4. **KV 提前释放或泄漏**：淘汰、完成、取消请求的生命周期交叉。
5. **batch 分段错位**：不同请求动态变化后仍按固定宽度切 logits。
6. **CUDA Graph 抖动**：宽度种类过多造成频繁 capture 或 eager fallback。
7. **峰值显存低估**：只按平均宽度估算，没有考虑最大宽度和 block 碎片。
8. **非确定性排序**：并列分数在不同设备上产生不同 parent/child 顺序。

## 8. 建议的落地顺序

第一阶段只让 \(B_d\) 动态，保持每父 beam 的候选数 \(K\) 为请求级固定值，并以 schedule 最大值作为候选输出容量。这样能够先验证搜索、fork 和 KV 生命周期，代价是窄层的 logits Top-K 输出仍有浪费。

第二阶段引入 ragged batch 和宽度 bucket，减少 padding，并验证多请求混合、EOS、取消、抢占和 prefix cache 命中场景。

第三阶段再让 \(K_d\) 动态，并改造 SamplingParams、logprob 输出 shape、kernel 和 remux 协议。这部分侵入性最大，不应与第一阶段同时推进。

## 9. 测试要求

至少覆盖：

- 宽度单调增加、单调减少和反复变化；
- 多个子 beam 共享同一父节点；
- schedule 比生成长度短或长；
- EOS、catalog 过滤造成有效候选不足；
- batch 内不同请求处在不同深度和宽度；
- 请求取消、抢占、KV swap/recompute；
- TP、DP 和单卡结果一致；
- graph bucket 与 eager 路径结果一致；
- 动态宽度全设为同一值时，与原固定宽度实现逐 token、逐分数一致；
- KV block 在请求结束后无泄漏。

## 10. 结论

Tree Attention 已经提供公共前缀 KV 共享及显式父子分支关系，因此实现动态 beam width 不需要改变 attention 的数学定义。最小改动是在每层全局 Top-K 后生成不同长度的 `fork_info`，并把真实活跃宽度传给下一步。

工程难点不在“每层取不同的 K”本身，而在固定宽度假设散布于候选 stride、batch 切片、KV 引用计数、内存预分配和 CUDA Graph shape 中。若现有系统的 Tree Attention 元数据已经是 ragged/offset 形式，整体属于中等改造；若 kernel、调度器和 graph 都将宽度固化，则属于高复杂度改造。

性能收益由各层宽度之和 \(\sum B_d\) 决定，显存容量则仍应按最大宽度与最坏分支生命周期规划。建议分阶段实现：先动态活跃宽度，再优化 ragged/bucket，最后才考虑动态每父候选数。

