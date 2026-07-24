# 动态 Beam Width：xAttention 与 Tree Attention 适配对比

## 1. 核心结论

**Tree Attention 明显比 xAttention 更适合动态 Beam Width。**

原因很直接：

- 动态 Beam Width 改变的是每层搜索树的节点数量；
- Tree Attention 本来就使用树结构表达不同分支；
- 每层保留多少 beam，只会改变下一层节点数量；
- 多个新 beam 可以直接共享同一个父节点的历史 KV；
- 不需要因为 beam 数变化而复制、压缩或重排完整历史 KV。

相比之下，当前 xAttention 方案建立在固定 `[Batch, Beam Width, ...]` 物理布局上。Beam Width 变化会影响输入 shape、KV 布局、Beam TopK、ACL Graph 和 BufferPool，因此改造范围更大。

简单判断：

| 方案 | 动态 BW 适配程度 | 核心功能改造难度 |
|---|---|---:|
| Tree Attention | 天然适配 | **低至中** |
| xAttention | 固定宽度布局，需要额外适配 | **中高至高** |

这里需要区分：

- **让 Tree Attention 支持动态 BW**：难度较低；
- **让整个在线服务支持异宽 batch、静态 Graph 和高吞吐**：仍有通用工程工作，但这不是 Tree Attention 本身造成的。

## 2. 动态 BW 对两种 Attention 意味着什么

设当前层有 `W_current` 个 beam，每个 beam 产生 `candidate_top_k` 个候选，下一层保留 `W_next` 个 beam：

```text
W_current 个父节点
    × candidate_top_k
    → 全局 Top-W_next
    → 下一层 W_next 个节点
```

两种方案在 Beam TopK 之前基本相同，差异发生在 TopK 之后：

```text
TopK 输出：
(parent_beam_id, token_id, score)
```

- xAttention 需要根据 `parent_beam_id` 重新整理下一步的物理 KV；
- Tree Attention 只需将其记录为新的父子关系。

## 3. Tree Attention 方案

### 3.1 为什么天然支持动态 BW

Tree Attention 将不同 beam 表示为一棵共享前缀的树：

```text
                 shared prefix
                      │
              ┌───────┴───────┐
           parent 0         parent 1
          ┌───┴───┐             │
       child 0  child 1       child 2
```

它不要求：

- 每层节点数相同；
- 每个父节点拥有相同数量的子节点；
- 所有叶子位于同一深度；
- beam width 在请求生命周期内保持不变。

动态 BW 只是让不同层的叶子数量发生变化：

```text
Depth 0: 1
Depth 1: 8
Depth 2: 32
Depth 3: 16
Depth 4: 4
```

这仍然是一棵普通的非规则树。

SpecInfer 已经使用 token tree 表示多条候选序列，并通过 tree-based parallel decoding 同时验证整棵树。[SpecInfer](https://arxiv.org/abs/2305.09781)

EAGLE-2 更直接地使用 context-aware dynamic draft tree，根据上下文动态改变树结构，并报告相对静态树 20%–40% 的额外加速。这说明 Tree Attention 可以自然承载动态节点数量和动态分支结构。[EAGLE-2](https://arxiv.org/abs/2406.16858)

DeFT 则专门面向具有共享前缀的树状推理，通过 prefix-aware KV 分区减少重复 KV IO，进一步说明树结构适合管理共享前缀和动态分支。[DeFT](https://arxiv.org/abs/2404.00242)

这些工作主要用于 speculative decoding，但它们证明的是同一个底层能力：**Tree Attention 可以处理节点数和分支数不固定的 token tree。**

### 3.2 需要修改什么

如果当前 Tree Attention 已经支持可变长度的 `fork_info`：

```text
fork_info = [
    (parent_id, token_id),
    ...
]
```

最小改造只有以下几项：

1. 增加 `beam_width_schedule[depth]`；
2. 当前层 TopK 使用 `W_next`，而不是固定 `beam_width`；
3. 根据入选候选生成长度为 `W_next` 的 `fork_info`；
4. 下一步设置：

```text
active_beam_width = len(fork_info)
```

Tree Attention 和 KV 侧继续按照已有 parent/child 关系工作。

核心流程：

```text
当前 W_current 个 beam
    ↓
生成候选
    ↓
全局保留 W_next 个
    ↓
构造 W_next 条 parent/child 关系
    ↓
下一步处理 W_next 个节点
```

### 3.3 扩宽与缩宽

扩宽：

```text
W_current=8 → W_next=32
```

多个 child 可以来自同一个 parent：

```text
parent 0 → child 0
parent 0 → child 1
parent 0 → child 2
```

这些 child 共享 parent 的历史 KV，只需要分别追加新的 token KV。

缩宽：

```text
W_current=32 → W_next=8
```

只保留 8 个入选 child。未入选分支从活跃集合移除，其 KV block 引用随正常分支回收流程释放。

因此，Tree Attention 不需要引入专门的“动态 BW KV 重排”。

### 3.4 Tree Attention 的实际难点

Tree Attention 的动态 BW 核心功能难度是**低至中**，主要确认以下事项：

| 检查项 | 难度 |
|---|---:|
| TopK 按层读取 `W_next` | 低 |
| `fork_info` 长度动态变化 | 低 |
| 多个 child 引用同一 parent | 低至中 |
| 淘汰分支 KV 引用释放 | 中 |
| EOS 与 active beam 分离 | 中 |

如果这些能力已经存在，动态 BW 主要是搜索策略层改动。

较复杂的情况是同一 batch 内请求宽度不同：

```text
request A: width=8
request B: width=32
request C: width=16
```

此时需要：

```text
beam_widths  = [8, 32, 16]
beam_offsets = [0, 8, 40, 56]
```

这会改动 scheduler、logits 分段和 batch metadata。但第一版可以要求同 batch 使用相同 schedule，从而避开这部分。

## 4. xAttention 方案

### 4.1 为什么改造更大

当前 xAttention 的固定 BW 通常体现在：

```text
Decode input [B × W]
Beam state   [B, W]
Unshared KV  [B, W, KVHeads, Steps, HeadDim]
TopK output  [B, W]
ACL Graph    固定 B × W shape
```

这里的 `W` 同时决定：

- 有效 beam 数；
- KV 的物理 beam 维；
- TopK 输出 shape；
- BufferPool 容量；
- Graph capture shape。

动态 BW 会同时打破这些固定假设。

### 4.2 需要修改什么

xAttention 至少需要增加：

```text
current_width
next_width
candidate_top_k
max_beam_width
active_beam_mask
```

物理张量建议仍按最大宽度分配：

```text
physical width = max_beam_width
valid width    = current_width
```

xAttention 不能再从 KV shape 推断真实 BW，只能读取显式 metadata。

### 4.3 KV 重排

TopK 后需要执行：

```text
new beam → parent beam → physical KV slot
```

缩宽时，需要将入选 KV 整理到下一步有效区域。

扩宽时，多个新 beam 可能来自同一个 parent，需要复制父 beam 的历史 KV。

物理 KV 重排成本随以下规模增加：

```text
Layers × Batch × W_next × HistoryLength × KVSize
```

这正是 xAttention 相比 Tree Attention 的主要劣势：Tree Attention 只增加父 KV 引用，xAttention 往往需要复制或重排父 KV 内容。

### 4.4 ACL Graph

动态宽度会产生多个运行 shape。通常需要使用 BW bucket：

```text
8、16、32、64、128
```

例如：

```text
logical width=48
physical bucket=64
```

多出的 16 个 beam 必须使用 mask 屏蔽，不能参与 attention 或 Beam TopK。

Graph cache key 至少要区分：

```text
(batch_bucket, beam_bucket, decode_step_bucket)
```

因此 xAttention 动态 BW 不只是修改 Beam TopK，还涉及 KV、算子 metadata、Model Runner、BufferPool 和 ACL Graph。

### 4.5 xAttention 的实际难点

| 改造项 | 难度 |
|---|---:|
| TopK 拆分 current/next width | 中 |
| 有效宽度与 mask | 中 |
| KV 收缩、扩展和重排 | 高 |
| 多 BW ACL Graph | 高 |
| 任意动态 shape | 高 |
| 同 batch 不同宽度 | 很高 |

所以 xAttention 整体属于**中高至高难度**。

## 5. 两种方案直接对比

| 对比项 | Tree Attention | xAttention |
|---|---|---|
| 对动态 BW 的适配性 | **天然适配** | 需要额外适配 |
| 宽度变化的本质 | 下一层树节点数变化 | 物理张量 beam 维变化 |
| 父子关系 | 原生表达 | 需要额外 parent mapping |
| 扩宽 | 多 child 共享 parent KV | 通常需要复制 parent KV |
| 缩宽 | 删除未入选分支 | 重排有效 KV |
| KV 历史复制 | 少 | 多 |
| 动态 BW 核心改造 | 搜索层 + 少量 metadata | 搜索、KV、算子、Graph |
| 固定 shape Graph | 仍需 bucket/padding | 必须重点改造 |
| 单请求动态 BW | 简单 | 较复杂 |
| 同 batch 异宽 | 中高难度 | 很高难度 |
| 总体难度 | **低至中** | **中高至高** |

## 6. 性能差异

设第 `d` 层宽度为 `B_d`，历史长度为 `L_d`。

Tree Attention 的有效 attention 工作量接近：

\[
O\left(\sum_d B_dL_dH\right)
\]

KV 增量空间接近：

\[
O\left(S_{prompt}+\sum_d B_d\right)
\]

xAttention 若按 bucket 执行，工作量接近：

\[
O\left(\sum_d \widehat{B_d}L_dH\right)
\]

其中 `\widehat{B_d}` 是向上取整后的物理 bucket。

xAttention 还存在历史 KV 物理重排成本，而 Tree Attention 主要增加 parent/child 和 block table 元数据。

因此，动态 BW 变化越频繁、生成历史越长，Tree Attention 的优势越明显。

## 7. 推荐实现方案

### 7.1 首选：直接基于 Tree Attention 实现

第一版建议：

```text
固定 beam_width_schedule
+ 同 batch 同宽
+ 固定 candidate_top_k
+ 动态 fork_info
+ Tree KV 共享
```

需要修改：

1. BeamSearchParams 增加 schedule；
2. 每层读取 `W_next`；
3. 全局 TopK 保留 `W_next`；
4. 生成动态长度的 `fork_info`；
5. 下一步使用真实 active width；
6. 验证淘汰、EOS 和取消场景的 KV 回收。

该版本不需要修改 Tree Attention 的核心数学逻辑，也不需要增加物理 KV 重排。

### 7.2 xAttention 作为兼容或备选路径

如果必须继续使用 xAttention：

```text
固定 schedule
+ 最大容量 BufferPool
+ current_width
+ active mask
+ BW Graph bucket
+ select_unshared_kv 物理重排
```

该路径可以工作，但改造量和运行开销都更大。

### 7.3 不建议第一版实现

第一版不建议同时加入：

- 同 batch 不同宽度；
- 任意宽度 Graph；
- 动态 `candidate_top_k`；
- 根据分数实时计算 BW；
- streaming refill。

这些属于调度和性能增强，不是验证 Tree Attention 动态 BW 所必需。

## 8. 实施难度修正

最终难度判断应为：

```text
Tree Attention 单请求动态 BW
    → 低难度

Tree Attention + 固定 schedule + 同 batch 同宽
    → 低至中难度

Tree Attention + 同 batch 异宽 + Graph 高性能
    → 中高难度

xAttention + 固定 schedule + KV 重排 + Graph bucket
    → 中高难度

xAttention + 任意异宽 batch
    → 高至很高难度
```

因此不能笼统地说“Tree Attention 动态 BW 难度很大”。准确说法是：

> Tree Attention 在数据结构和 KV 共享机制上天然支持动态 Beam Width。基础功能改造较小；复杂度主要来自异宽 batch 和静态 Graph 等外围执行系统。

## 9. 备选动态策略

以下方案只决定每层宽度如何产生，不改变 Tree Attention 更适合作为执行基础的结论：

- 分数阈值动态宽度：[Beam Search Strategies for Neural Machine Translation](https://arxiv.org/abs/1702.01806)
- Context-aware 动态树：[EAGLE-2](https://arxiv.org/abs/2406.16858)
- Dynamic-width speculative beam：[DSBD](https://arxiv.org/abs/2409.16560)
- Best-first 搜索：[Best-First Beam Search](https://arxiv.org/abs/2007.03909)

这些策略后续都可以输出每层 `W_next`，再由 Tree Attention 执行动态树。

## 10. 最终建议

动态 Beam Width 应优先基于 Tree Attention 实现，不应优先在固定布局的 xAttention 上做大规模动态化改造。

推荐路线：

```text
第一步：Tree Attention + 固定 schedule + 同 batch 同宽
第二步：增加 Graph bucket 和 padding mask
第三步：支持同 batch 异宽
第四步：接入分数或上下文自适应 BW 策略
```

如果现有 Tree Attention 的 `fork_info`、block table 和 KV 引用管理已经成熟，第一步主要是 Beam Search 层的修改，整体不属于高难度改造。
