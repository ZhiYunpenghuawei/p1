# Issue #37 EngineCore Beam Batch 入口 — 改动评审文档

> 状态：待评审
> 关联 issue: [#37](https://github.com/zhanghanleo10/vllm-gr/issues/37)
> 父 RFC: [#35](https://github.com/zhanghanleo10/vllm-gr/issues/35)
> 契约依赖: PR #292（`vllm_gr/v1/beam/batch_contracts.py`，已合入）
> 本文定位：解释本次代码改动的逻辑、文件职责，以及后续子任务与当前代码的对接点。

## 1. 背景与范围

#37 是 RFC #35 的 2/10 子任务，目标一句话：

> Online/Offline 共用的 EngineCore Beam Batch 入口：原子构造 B 个 native
> child `Request` 和一个 `BeamBatch`，只注册一个 FCFS 队列项，并把唯一一个
> 最终结果路由回调用方。

本次改动完成了 #37 的以下交付物：

- `EngineCoreBeamBatchIngress`：submit / cancel / final 路由；
- native child builder 与 `BeamBatch` 宿主对象；
- 最小 registry 与 terminal 去重（按 `batch_id + execution_id`）；
- 原子注册与 enqueue 失败 rollback；
- 前端统一入口 `beam_batch`（离线 `GRLLM` / 在线 `AsyncLLM`）；
- wire 消息 `BEAM_BATCH_SUBMIT=0x08`、`BEAM_BATCH_ABORT=0x09` 及 EngineCore
  分发；
- 单测 9 个 + patch 清单登记（GR-PATCH-016）。

不在本次范围：#39 Scheduler all-or-none 准入、#40 生命周期、#41 输出
schema/跨进程回传、#43 Worker run-to-completion。

## 2. 架构一句话

`beam_batch.py` 里的类是**被调用的组件**，不是 patch；patch 的是把它们接进
vLLM 的**接线层**。前端负责创建 `EngineCoreRequest` 并打包成
`BeamBatchRequest`；EngineCore 侧由 patched 的 socket 分发逻辑把消息送进
ingress。

## 3. 改动文件清单

| 文件 | 角色 | 新增关键内容 |
| --- | --- | --- |
| `vllm_gr/v1/engine/beam_batch.py` | 核心逻辑 | `BeamBatch`、`BeamBatchRecord`、`BeamBatchHandle`、`default_child_builder`、`build_beam_batch_request`、`EngineCoreBeamBatchIngress` |
| `vllm_gr/v1/engine/wire.py` | 协议编号 | `BEAM_BATCH_SUBMIT=0x08`、`BEAM_BATCH_ABORT=0x09` |
| `vllm_gr/v1/engine/codec.py` | 消息解码 | `beam_batch_submit` decoder、`decode_engine_core_payload` 新分支 |
| `vllm_gr/v1/engine/core.py` | EngineCore 侧装配与分发 | `_ensure_beam_batch_ingress`、`_handle_beam_batch_submit/abort`、`process_input_sockets` 新分支、`_enqueue_beam_batch` / `_abort_beam_batch`（seam） |
| `vllm_gr/v1/engine/engine_core_patch.py` | patch 安装 | patched `__init__` 装配 ingress；`apply_engine_core_request_patches` 绑定 handler |
| `vllm_gr/v1/engine/core_client.py` | 前端发送方法 | AsyncMP / SyncMP / Inproc 三套 submit / abort |
| `vllm_gr/v1/engine/core_client_patch.py` | 前端接线 | `apply_engine_client_patches`、`apply_async_llm_patches`（`AsyncLLM.beam_batch`） |
| `vllm_gr/v1/engine/async_llm.py` | 在线 facade | `beam_batch_fn` |
| `vllm_gr/entrypoints/gr.py` | 离线 facade | `GRLLM.beam_batch` |
| `tests/test_beam_batch_ingress.py` | 单测 | 9 个用例 |
| `docs/patch_inventory.yaml` / `docs/runtime_patch_inventory.md` | 清单 | GR-PATCH-016 |

## 4. 核心逻辑说明

### 4.1 `build_beam_batch_request`

输入 B 个 `EngineCoreRequest` + `BeamBatchParams` + `BeamOutputOptions`。
按顺序给每个 request 编 `item_index`，生成 `batch_id`，组装成不可变
`BeamBatchRequest`，跑 `validate_beam_batch_request` 契约校验，通过后返回。

### 4.2 `default_child_builder`

遍历 `BeamBatchRequest.items`，对每个 item 的 `EngineCoreRequest` 调用
`Request.from_engine_core_request(item.request, block_hasher)` 生成 native
`Request`，收集成 tuple，连同原 request 放进 `BeamBatch` 返回。

### 4.3 `EngineCoreBeamBatchIngress.submit_batch`

顺序：

1. 校验 `BeamBatchRequest` 与 `BeamBatchLimits`；
2. 调 `child_builder` 构造 `BeamBatch`（失败即抛错，此时未登记）；
3. 加锁：混合模式检查（Beam runtime 开启且存在普通请求时拒绝）、
   `batch_id` 重复检查；
4. 分配 `execution_id`，创建 `Future`，把 `BeamBatchRecord` 同时写入
   `_records_by_batch_id` 与 `_records_by_execution_id`；
5. 调 `enqueue` 回调；失败则回滚两条 registry 记录、标 FAILED、future 置
   异常、抛 `BeamBatchIngressError`；
6. 成功则 `enqueued=True`，返回 `BeamBatchHandle`。

线性化点：验证 → B 个 child 构造 → registry 登记 → 单次 enqueue，全部成功
才返回 handle。

### 4.4 `cancel`

按 `execution_id` 找 record；找不到返回 False；COMPLETE / CANCELLED 返回
False；否则置 CANCELLED、调 `cancel` 回调、取消 future、返回 True。

### 4.5 `route_terminal_result`

校验 `BeamBatchResult`；按 `batch_id` 找 record；找不到抛错；record 已
CANCELLED / FAILED 抛错；`terminal` 已非空抛重复完成错误；否则写
`terminal`、置 COMPLETE、`future.set_result(result)`、返回 handle。

### 4.6 `fail_beam_batch`

用 `batch_id` + 错误码构造 `failure` 结果，然后走 `route_terminal_result`
同一套完成逻辑。

## 5. 端到端调用顺序

### 5.1 离线默认（InprocClient，同进程）

```text
GRLLM.beam_batch(prompts, ...)
  → input_processor.process_inputs() × B     创建 EngineCoreRequest
  → build_beam_batch_request()               打包 BeamBatchRequest
  → InprocClient.submit_beam_batch()         同进程直调
  → ingress.submit_batch()                   核心逻辑
  → handle.result()                          阻塞等 final
```

### 5.2 在线 / 离线多进程（AsyncMPClient / SyncMPClient）

```text
AsyncLLM.beam_batch(...)
  → input_processor.process_inputs() × B
  → build_beam_batch_request()
  → AsyncMPClient.submit_beam_batch_async()  _send_input(BEAM_BATCH_SUBMIT)
  → ZMQ → EngineCoreProc.process_input_sockets()
  → decode_engine_core_payload()             按类型解码
  → _handle_beam_batch_submit()
  → ingress.submit_batch()
```

### 5.3 取消

```text
Inproc：ingress.cancel(execution_id)
MP：    _send_input(BEAM_BATCH_ABORT, batch_id)
        → process_input_sockets 分支 → _handle_beam_batch_abort → ingress.cancel(...)
```

### 5.4 patch 安装顺序

```text
1. vLLM general plugin 加载 → vllm_gr.plugin.register → re_apply_patches
2. 前端：patch_batch_and_fork() → 绑定三个 client 方法与 AsyncLLM.beam_batch
3. EngineCore 进程：run_engine_core() wrapper
     → apply_engine_core_request_patches()
         → 注册 wire 0x08/0x09
         → 包装 EngineCore.__init__ / EngineCoreProc.__init__
         → 绑定 handler、替换 process_input_sockets
4. EngineCore 构造：_ensure_beam_batch_ingress(self) 创建 ingress 实例
```

## 6. 当前代码与后续工作的对接点（重点）

这是评审最需要看的部分：后续子任务分别要在哪些函数、哪些数据结构上继续。

### 6.1 #39 Scheduler（all-or-none 准入）

- 对接点：`core.py::_enqueue_beam_batch`（当前只打日志）。
- 现状：#37 保证「一个 batch 只 enqueue 一次」，但 `BeamBatch` 还没真正放进
  `Scheduler.waiting`。
- 后续要做的：把 `record.batch`（`BeamBatch`，内含 B 个 native child
  `Request`）作为一个 FCFS 队列项加入 waiting；B 个 child 不独立入队。
  需要决定 `BeamBatch` 对象如何满足 waiting 队列的接口（或 patch Scheduler
  的排队逻辑）。
- 数据依赖：`BeamBatchRecord.batch` / `BeamBatch.children` 就是 #39 的输入。

### 6.2 #40 Lifecycle（cancel / failure / poison / exactly-once）

- 对接点：`core.py::_abort_beam_batch`（当前只打日志）；
  `beam_batch.py::cancel` / `route_terminal_result` / `fail_beam_batch`。
- 现状：ingress 状态机完整（WAITING → ACTIVE → COMPLETE / CANCELLED /
  FAILED），但"转发到 Scheduler/Worker"是占位。
- 后续要做的：active cancel 在 Worker 返回边界生效；terminal 提交后清理
  registry（当前 record 完成/取消后仍留在两个字典里，无回收逻辑）；
  cleanup 与 PromptKV/RunSlot 释放 exactly-once。
- 注意：`route_terminal_result` 对 CANCELLED / FAILED 的 record 拒绝 terminal，
  这是 #40 语义的一部分，后续需要确认与"pending cancel wins before commit"
  的时序是否一致。

### 6.3 #41 Output（final-only 结果与跨进程回传）

- 对接点：`beam_batch.py::route_terminal_result`；`BeamBatchResult` /
  `BeamOutputSequence` 契约（#292）。
- 现状：final 路由在 EngineCore 进程内完整；但结果没有跨进程送回前端。
  在线 `AsyncLLM.beam_batch` 目前只返回 `batch_id`。
- 后续要做的：EngineCore 在 `route_terminal_result` 后把最终结果写入
  output 通道（`EngineCoreOutputs` 或新增输出类型），前端 output processor
  按 `batch_id` 完成一个前端侧 future / queue；#41 同时定稿 logprobs /
  result schema。

### 6.4 #42 E2E

- 对接点：`gr.py::GRLLM.beam_batch`、`async_llm.py::beam_batch_fn`。
- 现状：离线默认 in-process 的提交路径可通（`handle.result()`），但真实
  Worker 还没产生 `BeamBatchResult`，所以 `route_terminal_result` 目前只在
  单测里被调用。
- 后续要做的：接上 #43 的 Worker 终态 → EngineCore 调
  `route_terminal_result`；在线补输出 adapter；补 E2E 用例。

### 6.5 #43 Worker（run-to-completion）

- 对接点：谁调用 `ingress.route_terminal_result(result)`——目前没有任何
  生产代码调用它，这是 #43 落地时的接线点。
- `BeamBatch.children` 是 Worker 做 B 个 child Full Prefill 的输入；
  `BeamBatchParams` / `BeamOutputOptions` 从 `BeamBatchRequest` 取。

### 6.6 #36 Contract 演进

- 对接点：`submit_batch` 已调用 `validate_beam_batch_request` 与
  `validate_beam_batch_request_limits`。后续契约字段变化只需改
  `batch_contracts.py`，ingress 无需改结构。

### 6.7 wire / client 扩展

- 0x08 / 0x09 已被本次占用；后续 #40 若需要新控制消息（如 lifecycle ack），
  从 `0x0A` 开始，沿用 `wire.py::_add_enum_member` 与 `codec.py` 的 decoder
  模式。
- client 方法遵循 `core_client_patch.py` 的既有模式：本体在
  `core_client.py`，绑定在 `apply_engine_client_patches` / 
  `apply_async_llm_patches`。

### 6.8 patch 体系

- GR-PATCH-016 已登记。后续 #39/#40 落地时，各自按 owner / installer /
  activation / validator / process_roles 更新 `docs/patch_inventory.yaml` 与
  `docs/runtime_patch_inventory.md`；若新增 validator，同步进
  `vllm_gr/patch.py`。

## 7. 测试覆盖

`tests/test_beam_batch_ingress.py`（9 个）：

- submit batch 只 enqueue 一次并路由 terminal；
- submit single 生成单 item batch；
- 重复 batch_id 拒绝；
- enqueue 失败 rollback（registry 清空）；
- mixed-mode fail-fast；
- 重复 terminal 拒绝；
- native child 物化（`BeamBatch.children`）；
- child 构造失败不登记；
- wire 0x08 / 0x09 注册。

## 8. 已知缺口与风险

1. `_enqueue_beam_batch` / `_abort_beam_batch` 是占位，真实 Scheduler /
   Worker 语义未验证（依赖 #39 / #40）。
2. `route_terminal_result` 没有生产调用方（依赖 #43）。
3. 跨进程 final 结果未回传；在线 facade 只返回 batch_id（依赖 #41 / #42）。
4. registry 无回收策略：COMPLETE / CANCELLED record 会一直留在内存
   （#40 清理）。
5. `GRLLM.beam_batch` 对 multiprocess 离线后端会显式报错，暂不支持。

## 9. 评审关注点

1. `submit_batch` 的线性化点（构造 → 登记 → enqueue → rollback）是否满足
   issue 验收标准？
2. `cancel` 的 CANCELLED 状态 + future 取消，与 #40 "pending cancel wins
   before commit" 的语义是否兼容？
3. `BeamBatchHandle` 持有可变 `record` 引用，跨进程场景下是否应改为只读
   快照 / 前端侧 future？
4. `_ensure_beam_batch_ingress` 在 `EngineCore.__init__` 之前装配，child
   builder 延迟读取 `self.request_block_hasher`，是否存在初始化时序风险？
5. wire 0x08 / 0x09 与 codec 的 decoder 是否需要补跨进程序列化/反序列化
   用例？
