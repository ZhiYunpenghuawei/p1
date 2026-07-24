# 动态 Beam Width 实现方案对比

## 1. 结论

动态 Beam Width 有两条主要适配路线：

1. **xAttention 方案**：保留规则张量和最大容量 BufferPool，通过有效宽度、KV 物理重排和 Graph bucket 支持宽度变化。
2. **Tree Attention 方案**：将 beam 表示为动态树，通过 parent/child 映射共享历史 KV，只维护实际活跃分支。

简单选择：

| 目标 | 推荐方案 |
|---|---|
| 尽快实现、兼容现有 ACL Graph 和固定张量 | xAttention |
| 减少 KV 复制、长期支持灵活动态宽度 | Tree Attention |
| 兼顾上线速度和长期性能 | xAttention 负责规则执行，Tree Attention 负责逻辑 KV |

推荐最终采用组合方案：

```text
xAttention：最大容量 + 有效宽度 + Graph bucket
Tree Attention：parent/child + 逻辑 KV 共享
```

## 2. 两种方案的共同改造

无论采用哪种 attention，都必须先拆分以下概念：

```text
current_width        当前参与 forward 的 beam 数
next_width           本层裁剪后保留的 beam 数
candidate_top_k      每个父 beam 产生的候选数
max_beam_width       最大物理容量
num_return_sequences 最终返回数量
```

每个 decode step 的公共流程为：

```text
W_current 个 parent
    × 每个 parent 的 candidate_top_k 个候选
    → 全局选择 W_next 个 beam
    → 得到 parent_id、token_id 和 score
    → 更新下一步的 beam 与 KV
```

扩宽必须满足：

\[
W_{next} \le W_{current}\times candidate\_top\_k
\]

需要共同修改的模块包括：

- Beam Width Schedule；
- Beam TopK 的输入父节点数和输出宽度；
- parent index 计算；
- sequence、score 和 EOS/completed 状态；
- scheduler 的当前活跃 token 数；
- Graph shape 或 Graph bucket；
- 动态宽度指标和正确性测试。

两种方案真正不同的地方，是入选新 beam 后如何组织和访问历史 KV。

## 3. xAttention 适配方案

### 3.1 核心思路

xAttention 继续使用规则的物理张量：

```text
物理容量 = max_beam_width
有效区域 = current_width
```

例如最大宽度为 128，当前宽度为 32：

```text
Beam scores [B, 128]       有效区域 [:, :32]
Sequence    [B, 128, T]    有效区域 [:, :32, :]
Unshared KV [B, 128, ...]  有效区域 [:, :32, ...]
```

xAttention 不能再从 KV 的物理 shape 推断真实宽度，metadata 必须显式携带：

```text
current_width
active_token_count
request/beam mapping
active_beam_mask
```

### 3.2 KV 更新

Beam TopK 输出：

```text
new_beam → parent_beam
```

通过 `select_unshared_kv` 按 parent index 物理整理历史 KV。

收缩：

```text
128 → 32
```

只保留 32 个入选路径，并把对应 KV 整理到下一步有效区域。

扩宽：

```text
32 → 64
```

多个新 beam 可能来自同一个 parent，需要复制该 parent 的历史 KV，然后分别追加新的后缀 KV。

物理重排开销近似随以下规模增长：

```text
Layers × Batch × W_next × DecodedSteps × KVSize
```

宽度变化越频繁、历史越长，重排成本越明显。

### 3.3 Graph 与调度

xAttention 方案适合使用有限 BW bucket：

```text
8、16、32、64、128
```

实际宽度向上匹配：

```text
W=48 → bucket=64
```

Graph key 至少应包含：

```text
(batch_bucket, beam_bucket, decode_step_bucket)
```

第一版建议同一 batch 内使用相同 schedule 和当前宽度，保持规则的 `[B, W, ...]` 布局。

### 3.4 优缺点

优点：

- 改造路径明确；
- BufferPool 地址稳定；
- 容易兼容固定 shape 的 ACL Graph；
- 第一版不需要引入 ragged layout；
- 适合快速上线。

缺点：

- 宽度切换需要物理 KV 重排；
- 扩宽时可能复制大量历史 KV；
- 按最大宽度分配时，常驻显存不会明显下降；
- bucket padding 会产生无效计算；
- 任意宽度和异宽 batch 支持较差。

### 3.5 实现难度

整体为**中高难度**。

最高风险集中在：

- 自定义 Beam TopK 算子能否区分 `candidate_top_k` 和 `next_width`；
- `select_unshared_kv` 能否支持动态输入/输出宽度和重复 parent；
- xAttention 是否支持有效宽度及 padding mask；
- ACL Graph 是否能按多个 BW bucket 稳定 replay。

## 4. Tree Attention 适配方案

### 4.1 核心思路

Tree Attention 不把每个 beam 看作完整、独立的 KV 序列，而是维护生成树：

```text
parent beam
    ├── child beam 1
    ├── child beam 2
    └── child beam 3
```

每次 Beam TopK 后只生成入选分支的映射：

```text
fork_info = [(parent_id, token_id), ...]
active_width = len(fork_info)
```

下一步 Tree Attention 直接读取实际活跃分支。attention 的数学定义不变，变化的是树结构和 KV block 映射。

### 4.2 KV 更新

扩宽时：

- 多个 child 可以引用同一个 parent 的历史 KV；
- 不需要立即复制完整历史 KV；
- 每个 child 只追加自己的新 token KV。

缩宽时：

- 未入选分支从活跃树中移除；
- 对应 KV block 引用计数递减；
- 没有其他活跃或保留分支引用时释放。

因此 Tree Attention 的核心收益是：

```text
保存 parent/child 关系
代替
反复复制和重排完整历史 KV
```

### 4.3 Batch 与调度

如果同一 batch 内请求宽度不同，需要使用 ragged metadata：

```text
beam_widths  [B]
beam_offsets [B+1]
总 beam 数   sum(beam_widths)
```

每一步都要同步更新：

- request offset；
- parent/child 映射；
- query 和 logits 分段；
- slot mapping；
- block table；
- 当前活跃宽度。

为了降低第一版复杂度，也可以暂时限制同 batch 同宽，并使用 Graph bucket。这样先获得逻辑 KV 共享收益，再逐步支持 ragged batch。

### 4.4 优缺点

优点：

- 动态扩宽和缩宽表达自然；
- 多个 child 可以共享 parent KV；
- 减少完整历史 KV 的物理复制；
- 实际 KV 占用更接近活跃树规模；
- 适合长期支持任意宽度和异宽 batch。

缺点：

- KV block 引用计数和生命周期复杂；
- scheduler、offset 和 block table 改造较大；
- ragged batch 容易出现跨请求分段错误；
- 动态 shape 会降低 Graph 复用率；
- 取消、抢占、EOS 和 early stop 的回收逻辑复杂。

### 4.5 实现难度

如果现有 Tree Attention 已有 `fork_info`、parent/child 和 block table 共享，整体为**中高难度**；如果这些能力尚未完整实现，则为**高难度**。

最高风险集中在：

- parent index 使用动态候选 stride 后是否正确；
- 多 child 共享同一 parent 时的 KV 引用；
- 被淘汰、完成和取消分支的 KV 回收；
- 多请求 ragged offset；
- Graph bucket 与逻辑宽度之间的 mask。

## 5. 核心对比

| 对比项 | xAttention | Tree Attention |
|---|---|---|
| 核心模型 | 最大容量 + 有效宽度 | 动态父子树 |
| KV 组织 | 每个 beam 的规则物理区域 | 公共路径共享 + 分支后缀 |
| KV 更新 | 物理重排/复制 | 逻辑 fork/引用 |
| 扩宽成本 | 复制父 beam 历史 KV | 增加父 KV 引用 |
| 缩宽成本 | 重排保留 KV | 删除分支并递减引用 |
| 常驻显存 | 通常按最大 BW 分配 | 更接近实际活跃树 |
| 张量布局 | 规则 `[B,W,...]` | ragged/offset 更自然 |
| Graph 兼容性 | 较好 | 一般，需要 bucket/padding |
| 同 batch 异宽 | 实现困难 | 可以支持，但改造较大 |
| 第一版开发量 | 较小 | 较大 |
| 长期扩展能力 | 一般 | 较好 |
| 主要性能风险 | KV 重排与 padding | metadata、调度与动态 shape |

## 6. 性能与复杂度对比

设第 \(d\) 层当前宽度为 \(B_{d-1}\)，下一层宽度为 \(B_d\)，每父候选数为 \(K_d\)，历史长度为 \(L_d\)。

两种方案共同的候选处理量为：

\[
O\left(\sum_{d=1}^{D} B_{d-1}K_d\right)
\]

### xAttention

模型 forward 的有效计算量近似为：

\[
O\left(\sum_{d=1}^{D}\widehat{B}_{d-1}L_dH\right)
\]

其中 \(\widehat{B}_d\) 是 bucket 后的物理宽度。

KV 重排还会增加近似成本：

\[
O\left(\sum_{d=1}^{D}
Layers \times B_d \times L_d \times KVSize
\right)
\]

若 BufferPool 按最大宽度分配，空间更接近：

\[
O\left(B_{max}\times D\right)
\]

### Tree Attention

Tree Attention 的计算量与实际活跃 beam 更接近：

\[
O\left(\sum_{d=1}^{D}B_{d-1}L_dH\right)
\]

理想 KV 增量空间近似为：

\[
O\left(S_{prompt}+\sum_{d=1}^{D}B_d\right)
\]

但会增加 \(O(\sum B_d)\) 规模的 parent/child、block table 和引用计数元数据。

### 直观判断

```text
宽度变化少、序列短、Graph 收益大
    → xAttention 更划算

宽度变化频繁、序列长、历史 KV 大
    → Tree Attention 更划算
```

## 7. 推荐组合方案

不建议把两种方案完全割裂。推荐：

```text
逻辑层：Tree Attention
    - parent/child
    - fork_info
    - KV block 引用
    - 淘汰分支回收

物理执行层：xAttention
    - 最大容量 BufferPool
    - current_width
    - BW bucket
    - active mask
    - ACL Graph replay
```

执行过程：

```text
Step d 使用 current_width 做 forward
    ↓
每个 parent 生成 K[d] 个候选
    ↓
全局选择 next_width 个 beam
    ↓
生成 parent/child 映射
    ↓
Tree 层更新逻辑 KV 引用
    ↓
xAttention 在匹配的物理 bucket 中执行下一步
```

这一方案的特点是：

- 用 Tree Attention 避免频繁复制完整历史 KV；
- 用 xAttention bucket 保持规则 shape 和 Graph 稳定；
- 逻辑宽度与物理宽度分离；
- padding beam 由 active mask 屏蔽；
- 第一版仍可限制同 batch 同宽。

## 8. 推荐实施顺序

### 第一阶段：xAttention 基础适配

- 增加固定 BW schedule；
- 拆分 current、next、max width；
- 拆分 `candidate_top_k` 和 `next_width`；
- 同 batch 同宽；
- 最大容量 BufferPool；
- 多 BW Graph bucket；
- 暂时使用 `select_unshared_kv` 物理重排。

目标是先验证动态宽度搜索语义和 Graph 路径。

### 第二阶段：接入 Tree KV

- Beam TopK 输出统一的 parent/child；
- 用 `fork_info` 维护分支；
- 使用 block table 引用代替完整 KV 复制；
- 完善淘汰、EOS、取消和 early stop 的引用回收；
- 保持同 batch 同宽，控制改造范围。

目标是减少宽度切换时的 KV 重排成本。

### 第三阶段：完整动态执行

- 同 batch 不同宽度；
- ragged `beam_widths/beam_offsets`；
- 动态 scheduler 分桶；
- 减少 padding；
- 支持动态 `candidate_top_k`；
- 根据指标决定是否保留物理 KV 重排 fallback。

## 9. 备选方案

以下方案可在主路径稳定后考虑，不作为当前主要实现：

| 备选方案 | 作用 | 与主方案的关系 |
|---|---|---|
| 分数阈值动态宽度 | 根据候选分差决定保留数 | 替代固定 schedule 的策略层 |
| DSBD | 上下文自适应宽度 + speculative forest | 需要 draft model，属于独立方向 |
| Best-First Beam Search | 优先扩展高价值候选 | 改变搜索顺序 |
| Streaming Variable-Width | batch 变窄后补充新请求 | 优化 scheduler 利用率 |

参考资料：

- [Beam Search Strategies for Neural Machine Translation](https://arxiv.org/abs/1702.01806)
- [Dynamic-Width Speculative Beam Decoding](https://arxiv.org/abs/2409.16560)
- [Best-First Beam Search](https://arxiv.org/abs/2007.03909)
- [A Streaming Approach for Efficient Batched Beam Search](https://aclanthology.org/2020.emnlp-main.366/)

## 10. 最终建议

如果只选一个短期方案，选择 **xAttention + 固定 schedule + BW bucket**，实现快且容易兼容现有执行链路。

如果考虑长期性能，选择 **Tree Attention 管理逻辑 KV，xAttention 管理物理执行**。这是更平衡的方案：

- xAttention 保证规则 shape、BufferPool 和 Graph 稳定；
- Tree Attention 解决扩宽、缩宽时的历史 KV 共享；
- 第一版通过“同 batch 同宽”限制复杂度；
- 后续再演进到 ragged batch 和自适应 BW。
