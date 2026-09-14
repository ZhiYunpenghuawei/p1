# Beam Search V1 A3 修改说明

## 1. A3 整体做了什么

A2 已经完成 Beam Search V1 的前端请求模型：

- 一个 Beam Session 对应一个原生 vLLM Request；
- 前端只提交一次完整的 `BeamBatchRequest`；
- 前端使用 `submit_once -> wait_final` 等待最终结果；
- Prefill 和所有 Decode 步骤不再由前端循环逐步提交。

A3 在此基础上，把“每个 Prefill/Decode Stage 如何调度”的控制权放到
EngineCore 和原生 vLLM Scheduler 内部。

整体调用链如下：

```text
A2 Frontend
    │ submit_gr_session(BeamBatchRequest)
    ▼
EngineCore Client
    │ SUBMIT_GR_SESSION
    ▼
EngineCore
    │ 原生 Request 只预处理、添加一次
    ▼
Native vLLM Scheduler + GRSessionState
    │
    ├─ 逻辑 Stage: Prefill -> Decode 0 -> Decode 1 -> ...
    └─ 物理 Dispatch: 每次下发一个唯一 dispatch_id
         │ SchedulerOutput.gr_stage_metadata
         ▼
A4 Worker（回显 session_id / dispatch_id / stage_index）
         │ ModelRunnerOutput.gr_worker_results
         ▼
Scheduler update：回收 Dispatch、推进逻辑 Stage、缓存 terminal result
         │ 已下发任务全部 drain
         ▼
EngineCore finalizer：完成原生 Request 并发布一次最终结果
         │
         ▼
A2 Frontend wait_final()
```

A3 最关键的变化是允许同一 Session 最多同时存在两个尚未被 Host 回收的
物理 Dispatch：

```python
GR_LOOKAHEAD_DEPTH = 2
```

也就是：

- 一个 Dispatch 正在 GPU 上执行；
- 下一个 Dispatch 已经由 CPU 调度并进入原生异步队列；
- CPU 不必等待上一个 Dispatch 的 Future 完成后才开始调度下一步；
- 逻辑 Stage 之间只传递轻量设备状态引用，不把中间 Beam 数据送回前端。

新架构明确区分两种编号：

- `stage_index` 表示会产生 Beam 输出的逻辑阶段，决定 token 预算和设备状态依赖；
- `dispatch_id` 表示一次真实下发，所有物理调度都唯一，包括初始 chunked Prefill
  和抢占恢复后的 Prefill 重计算。

不产生输出的 Prefill chunk 可以复用同一个 `stage_index`，但必须使用新的
`dispatch_id`。因此，逻辑阶段计数不会被重计算消耗，而每次物理完成仍可被精确、
幂等地回收。

Stage 间的依赖表示为：

```text
GRDeviceStateRef(session_id, stage_index)
```

它只是一个逻辑 Stage 引用，不等同于 `dispatch_id`，也不包含真实 Tensor。
真正的 CUDA Buffer、Stream 和 Event 依赖由后续 A4 实现。

## 2. A3 的设计边界

### 2.1 A3 负责的内容

- 新增一次性提交完整 Session 的 EngineCore wire request；
- 为同步、异步多进程和 in-process 模式增加提交入口；
- 在原生 Scheduler 上维护持久的 GR Session 状态；
- 分离逻辑 Stage 编号和物理 Dispatch 编号；
- 实现深度为 2 的 bounded look-ahead；
- 维护逻辑 issued/completed、物理 in-flight 和 output placeholder 计数；
- 把设备状态依赖附加到 `SchedulerOutput`；
- 处理初始 chunked Prefill、抢占重计算、正常结束、提前结束、取消和失败；
- 校验 Worker 返回身份和载荷，并把协议错误限制在单个请求内；
- 缓存终态结果，等待所有已下发任务排空后只发布一次；
- 为 A4 Worker result 和 A5 terminal result 预留传输通道；
- 给 `LLMEngine` 和 `AsyncLLM` 安装 submit/wait/abort/release 接口。

### 2.2 A3 不负责的内容

- A4 才负责持久化 GPU Beam State 和真正执行设备依赖；
- A5 才负责最终 Beam 选择、D2H 和 `BeamBatchResult`；
- A6 才负责完整离线/在线输出转换和端到端接线；
- A3 不实现新的模型 Kernel；
- A3 不创建第二套 Scheduler；
- Phase A 当前只支持一个 Session 中一个原生 Request，即 `B=1`；
- A3 不安全地扩大 look-ahead 深度。

## 3. 文件修改总览

纯 A3 相对合入后的 A2 修改 11 个文件：

| 文件 | 作用 |
| --- | --- |
| `vllm_gr/v1/engine/beam_search_v1.py` | 定义 Session、逻辑 Stage、物理 Dispatch、设备状态和 Worker result 契约。 |
| `vllm_gr/v1/engine/gr_async_scheduler.py` | 实现 A3 Session 状态机、bounded look-ahead、Worker 校验和终态排空。 |
| `vllm_gr/v1/engine/gr_session_client.py` | 连接 A2 frontend 与 A3 EngineCore。 |
| `vllm_gr/v1/engine/wire.py` | 注册 `SUBMIT_GR_SESSION=0x0a`。 |
| `vllm_gr/v1/engine/codec.py` | 将新 wire payload 解码为 `BeamBatchRequest`。 |
| `vllm_gr/v1/engine/core.py` | 在 EngineCore 内一次性接收和预处理 Session，并通过请求错误通道返回配置失败。 |
| `vllm_gr/v1/engine/core_client.py` | 实现异步多进程 Session 提交。 |
| `vllm_gr/v1/engine/core_client_patch.py` | 给各种 Client 和 frontend 安装新方法。 |
| `vllm_gr/v1/engine/engine_core_patch.py` | 把 A3 状态机嵌入原生 Scheduler 生命周期，并负责原生 Request 的终态发布。 |
| `tests/test_gr_async_scheduler.py` | 测试 look-ahead、重计算、终态 drain、错误隔离和 abort。 |
| `tests/contract/test_engine_core_protocol_contracts.py` | 测试新 wire value 和协议清单。 |

## 4. `beam_search_v1.py`：核心数据模型和接口

这个文件主要定义数据结构和接口，不直接执行 Scheduler 逻辑。

### 4.1 `GRStageKind`

定义 Stage 类型：

- `PREFILL`；
- `DECODE`。

Stage 0 固定是 Prefill，Stage 1 开始是 Decode。

### 4.2 `GRSessionStatus`

定义 Session 生命周期：

- `WAITING`：Session 已注册，尚未开始；
- `RUNNING`：至少有一个 Stage 已发出；
- `FINISHING`：不再发新 Stage，但仍需回收已发出的任务；
- `FINISHED`：所有任务都已回收；
- `ABORTED`：用户取消；
- `FAILED`：执行失败。

典型状态转换：

```text
WAITING -> RUNNING -> FINISHING -> FINISHED
                  \-> ABORTED
                  \-> FAILED
```

### 4.3 `GRDeviceStateRef`

表示 Worker 设备状态的逻辑引用：

```python
GRDeviceStateRef(
    session_id="session-0",
    stage_index=1,
)
```

它不携带 token、parent、score 或 Tensor，只描述某个 Stage 的设备输出属于
哪个 Session 和哪个 Stage。

例如 Stage 2 的 `input_device_state` 应当引用 Stage 1 的
`output_device_state`。

### 4.4 `GRSessionState`

每个 Beam Session 对应一个 `GRSessionState`，它是 A3 的核心控制面状态。

主要字段：

| 字段 | 含义 |
| --- | --- |
| `session_id` | EngineCore 内部 Session ID。 |
| `request` | A2 构造的完整 `BeamBatchRequest`。 |
| `max_decode_steps` | Decode Stage 数量。 |
| `status` | 当前 Session 生命周期状态。 |
| `issued_stages` | 已经调度出去的逻辑 Stage 数。 |
| `completed_stages` | 已从 Worker output 确认完成的 Stage 数。 |
| `output_placeholders` | 原生 Request 为未确认输出保留的位置数。 |
| `confirmed_token_count` | 已确认产生的 token 数。 |
| `pending_device_token_count` | 已调度但尚未在 Host 确认的 token 数。 |
| `latest_device_state` | 最近一次发出 Stage 的设备状态引用。 |
| `terminal_result_pending` | 是否仍在等待最终结果安全落地。 |
| `_next_dispatch_id` | 下一次物理下发使用的单调递增编号。 |
| `_in_flight_dispatches` | 尚未回收的物理 `dispatch_id` 集合。 |
| `_pending_terminal_result` | drain 期间缓存的最终 `BeamBatchResult`。 |
| `_terminal_emitted` | 最终结果是否已经发布，用于保证 exactly-once。 |

核心计数不变量：

```text
0 <= completed_stages <= issued_stages <= total_stages
```

属性：

- `total_stages`：总逻辑 Stage 数；
- `in_flight_stages`：`issued_stages - completed_stages`，只表示尚未确认的输出型逻辑
  Stage，不用于限制物理队列深度；
- `next_stage_kind`：判断下一步是 Prefill、Decode，还是已经没有下一步。

物理调度深度由 `len(_in_flight_dispatches)` 判断。逻辑计数与物理集合必须分开，
因为抢占重计算会产生新的物理 Dispatch，但不产生新的逻辑 Stage。

有效 Beam token 预算包含 Prefill 产生的第一个 token，因此：

```text
total_stages = beam_params.max_tokens
max_decode_steps = beam_params.max_tokens - 1
```

### 4.5 `GRStageMetadata`

描述一次原生 SchedulerOutput 中对应的 GR Stage。

主要字段：

- `session_id`：Session 标识；
- `dispatch_id`：本次物理下发的唯一编号；
- `stage_index`：逻辑 Stage 编号；
- `kind`：Prefill 或 Decode；
- `decode_step`：Decode 的零基编号；
- `is_last_stage`：是否是预算中的最后一个 Stage；
- `input_device_state`：本 Stage 消费的设备状态；
- `output_device_state`：本 Stage 将产生的设备状态；
- `beam_params`：Beam 宽度、温度、最大 token 等参数；
- `output_options`：返回序列数以及是否包含分数、parent 等选项；
- `num_prompt_tokens`：原生 Scheduler 看到的权威 Prompt 长度；
- `produces_output`：本次调度是否真正产生逻辑 Beam 输出；
- `session_init`：为后续 Worker 初始化预留的可选请求字段。

实际挂载位置是：

```python
SchedulerOutput.gr_stage_metadata
```

其中 `session_init` 当前在纯 A3 的 `attach_gr_stage_metadata()` 中没有填充，
是给后续阶段保留的契约字段。

### 4.6 `GRWorkerResult`

定义后续 A4 Worker 返回给 A3 的控制面结果：

- `session_id`；
- `dispatch_id`；
- `stage_index`；
- `produced_token_count`；
- `device_state`；
- `finished`；
- `terminal_result`。

中间 Stage 只返回控制信息和设备引用，不应把完整 Beam Token、Parent 和
Score 传回前端。A4 返回结果时必须原样回显 metadata 中的 `session_id`、
`dispatch_id` 和 `stage_index`，A3 使用这组三元身份拒绝错配结果。

`produced_token_count` 只能为 0 或 1；0 只允许出现在终态结果中。Worker 一旦
提供 `gr_worker_results` 映射，本 Session 的 entry 就是必需的。整个属性缺失时，
才走纯 A3、尚未接入 A4 的兼容路径。

### 4.7 `GRStageCompletion`

A3 消费一次 Worker 输出后形成的统一完成记录，包括：

- 完成的 Session 和 Stage；
- 是否结束；
- 已确认 token 数；
- 最新设备状态；
- 原始 `ModelRunnerOutput`；
- 错误信息。

### 4.8 `GRAsyncSchedulerInterface`

这是一个 `Protocol`，描述 Scheduler 被 patch 后应该具备的方法：

- `add_gr_session()`；
- `can_issue_gr_stage()`；
- `reserve_gr_output_placeholder()`；
- `attach_gr_stage_metadata()`；
- `update_gr_from_output()`；
- `finish_gr_session()`。

它只定义接口，具体实现位于 `gr_async_scheduler.py`。

### 4.9 `GRModelExecutorInterface`

描述 A4 需要满足的 ModelExecutor 调用约定：

```python
execute_model(
    scheduler_output,
    non_block=True,
)
```

非阻塞执行时返回 `Future[ModelRunnerOutput]`。

### 4.10 `GREngineCoreInterface`

描述 EngineCore 侧的 GR 接口：

- `submit_gr_session()`：一次性提交 Session；
- `step_with_batch_queue()`：沿用原生异步 EngineCore 循环；
- `finish_gr_session()`：完成 drain 后取得终态结果；
- `abort_gr_session()`：取消 Session。

这些也是接口约定，不是新的 EngineCore 类。

## 5. `gr_async_scheduler.py`：A3 状态机实现

这是 A3 最主要的新增实现文件。

### 5.1 常量

`GR_STAGE_OUTPUT_ATTR = "gr_stage_metadata"`：

- SchedulerOutput 上的 GR Stage metadata 字段。

`GR_WORKER_RESULT_ATTR = "gr_worker_results"`：

- A4 Worker 在 ModelRunnerOutput 上附加 Stage result 的字段。

`GR_BATCH_RESULT_ATTR = "gr_batch_results"`：

- A5 在 ModelRunnerOutput 上附加最终 `BeamBatchResult` 的字段。

`GR_LOOKAHEAD_DEPTH = 2`：

- 同一 Session 最多允许两个未回收的物理 Dispatch。

### 5.2 `_sessions()`

延迟创建并返回：

```python
scheduler._gr_v1_sessions
```

其结构为：

```text
session_id -> GRSessionState
```

### 5.3 `add_gr_session()`

注册新的 Beam Session。

主要工作：

1. 检查 `async_scheduling=True`；
2. 检查当前只支持 `B=1`；
3. 检查 `batch_id == EngineCoreRequest.request_id`；
4. 防止 Session ID 重复；
5. 创建 `GRSessionState`；
6. 计算 `max_decode_steps = max_tokens - 1`；
7. 保存到 `scheduler._gr_v1_sessions`。

### 5.4 `can_issue_gr_stage()`

判断是否允许继续发下一个 Stage。

要求：

- Session 状态为 `WAITING` 或 `RUNNING`；
- 还有剩余逻辑 Stage；
- `len(session._in_flight_dispatches) < GR_LOOKAHEAD_DEPTH`。

它是 bounded look-ahead 的核心判断函数。这里不能使用
`issued_stages - completed_stages`，否则不产生输出的 Prefill 重计算不会占用队列
槽位，可能突破真实的异步队列深度。

### 5.5 `reserve_gr_output_placeholder()`

增加：

- `output_placeholders`；
- `pending_device_token_count`。

它表示某个结果已经被调度，但尚未由 Host 确认。

当前主路径还会从原生 Request 的 `num_output_placeholders` 同步占位信息；
这个函数同时作为 Scheduler 的显式扩展接口保留。

### 5.6 `attach_gr_stage_metadata()`

原生 Scheduler 完成一次调度后，为真正被选中的 GR Request 构造
`GRStageMetadata`。

主要行为：

1. 判断 Session 是否出现在本次 `SchedulerOutput`；
2. 获取原生 Request，并直接依据 `native_request.is_prefill_chunk` 判断本次是否为
   不产生输出的 Prefill；
3. 为每次真实下发分配唯一 `dispatch_id`，立即加入
   `_in_flight_dispatches`；
4. 计算逻辑 `stage_index` 和 Prefill/Decode 类型；
5. 为 metadata 构造预期的 `output_device_state`，并把上一个逻辑 Stage 的状态放入
   `input_device_state`；
6. 写入 Beam 参数、输出参数和 Prompt 长度；
7. 仅对输出型 Stage 更新 `issued_stages`、`status`、
   `latest_device_state` 和逻辑 placeholder 状态；
8. 同步原生 output placeholder 计数。

判断 Prefill 时不再要求 `issued_stages == 0`。请求被抢占后，原生 Scheduler 会
重置 computed token 并再次发出 Prefill chunk；这些 Dispatch 必须标记为
`PREFILL`、`produces_output=False`，保持原来的 `stage_index`，且不能发布新的设备
`latest_device_state`。

### 5.7 `update_gr_from_output()`

消费 Worker 完成结果，是物理 Dispatch 回收点，也是输出型逻辑 Stage 的
`completed_stages` 推进点。

主要行为：

1. 获取本次 Stage metadata；
2. 按 `dispatch_id` 检查幂等性：不在 `_in_flight_dispatches` 中的重复或过期结果
   直接忽略；
3. 从集合中回收本次物理 Dispatch，重新开放一个 look-ahead 槽位；
4. 对 `produces_output=False` 的 Prefill 重计算只完成物理回收，不推进逻辑计数；
5. 读取并校验 `ModelRunnerOutput.gr_worker_results` 和
   `gr_batch_results` 映射；
6. 严格校验 `session_id`、`dispatch_id`、`stage_index`、token 数、设备状态引用和
   terminal result；
7. 对正常输出更新 completed、confirmed token、pending token 和 placeholder；
8. Worker 报告 finished 或 Stage 预算耗尽时进入 `FINISHING`，缓存终态结果，
   停止发新 Stage；
9. 已经进入 `FINISHING`、`ABORTED` 或 `FAILED` 后，后继 Dispatch 的完成只负责
   drain，不再确认 token、覆盖设备状态或替换终态结果。

`update_gr_from_output()` 不直接完成原生 Request。正常终态结果和发布所有权交给
`engine_core_patch.py`：只有物理 Dispatch 全部 drain 后，才能完成原生 Request、
清除剩余占位、发布一次最终 output 并删除 Session。外部 abort/error 路径不发布
缓存的正常终态，可在物理任务 drain 后直接清理 Session。

Worker 协议错误由内部 `_GRWorkerProtocolError` 表示，但异常不会越过 Scheduler
output 边界。`_fail_gr_dispatch()` 会把它转换为当前请求的
`BeamBatchFailure(code="worker_protocol_error")`，进入相同的终态 drain 流程，
不会杀死 EngineCore 线程或影响其他请求。

### 5.8 Worker 载荷和失败辅助函数

- `_result_mapping()`：统一读取 Worker 扩展属性，并拒绝非 Mapping 载荷；
- `_read_worker_payload()`：恢复、校验 `GRWorkerResult` 和 `BeamBatchResult`，执行身份、
  token 数及设备引用检查；
- `_sync_output_placeholders()`：优先从仍存活的原生 Request 同步 placeholder，原生
  Request 已退出时回退到 Session 计数；
- `_make_failure_result()`：构造请求级 `BeamBatchFailure`；
- `_fail_gr_dispatch()`：把 Worker 协议异常转为终态 failure，并继续执行正常 drain。

### 5.9 终态辅助函数

- `_gr_session_ready_to_finalize()`：要求状态为 `FINISHING`、存在待发布终态、尚未
  发布，且 `_in_flight_dispatches` 已为空；
- `_gr_terminal_result()`：返回缓存的终态结果；若异常路径没有结果，则构造结构化
  failure 作为保护；
- `_close_gr_session()`：设置 exactly-once 标志并从 Session 表中移除状态。

### 5.10 `finish_gr_session()`

显式设置 Session 的终态。

当状态为 `FAILED` 时，返回结构化失败结果：

```text
failure.code = "engine_failure"
```

它只产生控制面错误，不生成真实 Beam 输出。

### 5.11 `abort_gr_session()`

取消 Session：

- 设置 `ABORTED`；
- 清除待发布的正常 terminal result，使外部取消优先；
- 调用原生 `scheduler.finish_requests()`；
- 使用 `RequestStatus.FINISHED_ABORTED`。

### 5.12 `release_gr_session()`

释放 Scheduler 中的 Session 状态。

如果 `_in_flight_dispatches` 非空，则拒绝释放，避免任务还在设备或队列中时提前
清理。

### 5.13 `install_gr_scheduler_methods()`

把这些独立函数动态安装到 vLLM `Scheduler` 类：

```python
Scheduler.add_gr_session
Scheduler.can_issue_gr_stage
Scheduler.reserve_gr_output_placeholder
Scheduler.attach_gr_stage_metadata
Scheduler.update_gr_from_output
Scheduler.finish_gr_session
Scheduler.abort_gr_session
Scheduler.release_gr_session
```

因此 A3 没有复制或派生第二套 Scheduler。

## 6. `gr_session_client.py`：前端 Session 适配

这个文件连接 A2 frontend 和 A3 EngineCore。

### 6.1 `SyncGRSessionHandle`

同步离线调用句柄，保存原始 `BeamBatchRequest`。

### 6.2 `AsyncGRSessionHandle`

异步在线调用句柄，保存：

- 原始 `BeamBatchRequest`；
- 原生 `RequestOutputCollector`。

### 6.3 `_terminal_result()`

从原生输出的以下位置读取最终结果：

```python
output.kv_transfer_params["gr_batch_result"]
```

如果 A4/A5 没有附加最终结果，则返回结构化失败：

```text
worker_result_unavailable
```

这样不会把原生占位 token 误认为真正的 Beam 输出。

### 6.4 `submit_gr_session_sync()`

同步提交流程：

1. 把原生 Request 注册到 `LLMEngine.output_processor`；
2. 调用 `engine.engine_core.submit_gr_session()`；
3. 返回 `SyncGRSessionHandle`。

### 6.5 `wait_gr_session_sync()`

循环调用 `engine.step()`，直到目标 Session 的输出 finished，然后提取最终
`BeamBatchResult`。

### 6.6 `abort_gr_session_sync()`

调用：

```python
engine.abort_request([session_id], internal=True)
```

这里必须使用内部 Session ID，不能按用户的外部 request ID 路由。

### 6.7 `release_gr_session_sync()`

同步 release 是幂等确认。Scheduler 在已发 Stage 全部 drain 后会自行清理。

### 6.8 `submit_gr_session_async()`

异步提交流程：

1. 创建 `RequestOutputCollector`；
2. 注册原生 Request；
3. 启动 AsyncLLM output handler；
4. 调用 `engine_core.submit_gr_session_async()`；
5. 返回 `AsyncGRSessionHandle`。

### 6.9 `wait_gr_session_async()`

等待 collector 输出，直到 `output.finished=True`，再提取最终
`BeamBatchResult`。

### 6.10 `abort_gr_session_async()`

调用：

```python
await engine.abort(session_id, internal=True)
```

### 6.11 `release_gr_session_async()`

与同步 release 一样，是幂等确认。

### 6.12 `install_gr_frontend_methods()`

把同步方法安装到 `LLMEngine`，把异步方法安装到 `AsyncLLM`：

```text
submit_gr_session
wait_gr_session
abort_gr_session
release_gr_session
```

## 7. `wire.py`：EngineCore wire 类型

A3 新增：

```python
SUBMIT_GR_SESSION_WIRE_VALUE = b"\x0a"
```

### 7.1 `_add_enum_member()`

动态向 vLLM 的 `EngineCoreRequestType` 添加成员，并检查：

- 名字冲突；
- wire value 冲突；
- 重复安装的幂等性。

### 7.2 `register_engine_core_request_types()`

注册所有 vLLM-GR request type。A3 在已有类型后新增：

```text
SUBMIT_GR_SESSION = 0x0a
```

## 8. `codec.py`：wire payload 解码

### 8.1 `EngineCoreDecoders`

保存各 request type 对应的 `MsgpackDecoder`。

A3 新增：

```python
submit_gr_session: MsgpackDecoder
```

### 8.2 `build_engine_core_decoders()`

为新请求创建类型化解码器：

```python
MsgpackDecoder(BeamBatchRequest)
```

因此 EngineCore 收到的是完整类型对象，而不是无类型 dict。

### 8.3 `decode_engine_core_payload()`

当 request type 是 `SUBMIT_GR_SESSION` 时，选择
`submit_gr_session` decoder。

## 9. `core.py`：EngineCore 接收入口

这个文件原本已有旧 Beam EngineCore 扩展。A3 主要新增一个处理函数和一个
dispatch 分支。

### 9.1 `_handle_submit_gr_session()`

一次性接收完整 Session。

处理流程：

1. 检查当前只支持 `B=1`；
2. 检查 `async_scheduling=True`；
3. 检查 EngineCore batch queue 至少有两个槽；
4. 调用原生 `preprocess_add_request()`，只预处理一次；
5. 将完整请求挂到原生 Request：

   ```python
   native_request.gr_batch_request = batch_request
   ```

6. 通过原生 add-request 路径把 Request 加入 EngineCore。

Scheduler 的 patched `add_request()` 会识别 `gr_batch_request` 并创建
`GRSessionState`。

`B=1`、`async_scheduling=True` 和 batch queue 至少两个槽位等后端约束都在
request-level `try` 内校验。失败时调用已有 `_handle_request_preproc_error()`，
把结构化错误返回给当前请求；配置不支持不会让异常逃出
`process_input_sockets()`、终止 EngineCore 输入线程并导致调用方永久等待。

### 9.2 `dispatch_gr_engine_core_input()`

A3 增加 `SUBMIT_GR_SESSION` 分支，将 payload 交给
`_handle_submit_gr_session()`。

`core.py` 中其他 Beam 函数大多属于已有路径，不是 A3 新增的主要逻辑。

## 10. `core_client.py`：异步多进程提交

### 10.1 `submit_gr_session_async()`

这是 `AsyncMPClient` 的 A3 提交实现。

主要工作：

- 从 `BeamBatchRequest` 取出原生 Request；
- 填充 `client_index` 和 `current_wave`；
- 选择目标 DP EngineCore；
- 发送 `SUBMIT_GR_SESSION`；
- 必要时发送 `FIRST_REQ`；
- 等待发送完成；
- 确保 EngineCore output queue task 已启动。

整个 Session 只发送一次，不会每个 Decode step 都发一次 frontend RPC。

## 11. `core_client_patch.py`：安装客户端和前端方法

### 11.1 `sync_mp_submit_gr_session()`

同步多进程客户端实现：

- 填充 client/wave 信息；
- 维护 DP engine-running 状态；
- 发送 `SUBMIT_GR_SESSION`。

### 11.2 `inproc_submit_gr_session()`

非多进程模式直接调用：

```python
self.engine_core._handle_submit_gr_session(batch_request)
```

### 11.3 `apply_request_type_patches()`

注册新的 EngineCore wire enum。

### 11.4 `apply_engine_client_patches()`

分别安装：

- `AsyncMPClient.submit_gr_session_async`；
- `SyncMPClient.submit_gr_session`；
- `InprocClient.submit_gr_session`。

### 11.5 `apply_async_llm_patches()`

A3 新增：

```python
install_gr_frontend_methods(LLMEngine, AsyncLLM)
```

用于安装 submit/wait/abort/release 前端方法。

### 11.6 `apply_batch_fork_patches()`

总安装入口，保证 frontend、client 和 EngineCore 子进程都具备对应 patch。

## 12. `engine_core_patch.py`：嵌入原生 Scheduler 生命周期

这是 A3 和原生 vLLM Scheduler 真正结合的地方。

### 12.1 `apply_engine_core_request_patches()`

A3 增加：

- `_handle_submit_gr_session` 方法绑定；
- EngineCore 和 EngineCoreProc 的新请求处理；
- 多进程和 in-process fallback dispatch。

### 12.2 `apply_scheduler_patch()`

这是 A3 最重要的集成函数。

第一步，调用：

```python
install_gr_scheduler_methods(Scheduler)
```

第二步，包装 `Scheduler.add_request()`：

- 检测 `request.gr_batch_request`；
- 调用 `add_gr_session()`；
- 然后继续走原生 add-request。

第三步，包装 `Scheduler.finish_requests()`：

- 使用原生方法返回的 `(request_id, client_index)` 列表同步 GR Session；
- 保留 `finish_requests(None, status)` 的 abort-all 语义；
- 不受生成器参数已被原生方法消费的影响；
- 区分内部 GR 终态发布和外部 abort/error；
- 外部终止覆盖已经缓存但尚未发布的正常终态，并防止继续发新 Stage。

第四步，包装 `Scheduler.schedule()`：

1. 遍历 GR Session；
2. 对达到 look-ahead 上限的 Request，临时将 `request.max_tokens=0`；
3. 调用原生 `Scheduler.schedule()`；
4. 为真正被选中的 GR Request 构造 `GRStageMetadata`；
5. 写到 `SchedulerOutput.gr_stage_metadata`；
6. 在 `finally` 中恢复原始 `max_tokens`。

临时设为 0 只表示“本轮不要再调度这个已饱和 Session”，不会永久修改生成预算，
也不会创建新的调度循环。其他普通 Request 仍由原生 Scheduler 正常选择。

### 12.3 `_sync_gr_sessions_after_native_finish()`

原生 `finish_requests()` 返回真正被终止的请求列表。A3 以这个返回值为唯一事实
来源更新 Session，而不是再次遍历调用参数。这样同时修复：

- `request_ids=None` 表示终止全部请求时遗漏 GR Session；
- `request_ids` 是生成器且已被原生方法消费时无法二次遍历；
- 未调度过、不会再收到 Worker completion 的 Session 无法清理。

`_GR_FINALIZING_IDS_ATTR` 是一个短生命周期的内部标记：

- `_finalize_ready_gr_session()` 主动完成原生 Request 时设置标记，保留已经缓存的
  GR terminal result；
- 用户 abort 或原生错误没有该标记，必须覆盖 pending terminal，取消或失败拥有
  更高优先级。

### 12.4 原生 Request 终态辅助函数

- `_find_gr_output()`：在原生 update 结果中查找当前 Session 的 output；
- `_suppress_native_terminal()`：Session 尚有物理 Dispatch 时，撤销原生过早产生
  的 finish 标记，继续等待 drain；
- `_finish_native_gr_request()`：清零推测性 placeholder，并带内部 finalizing 标记
  调用原生 `finish_requests()`；
- `_discard_pending_native_finish()`：避免下一次 update 重复发布同一个 finished ID；
- `_finalize_ready_gr_session()`：在 drain 完成后，把缓存结果附加到现有 output；若
  本轮没有 output，则合成一个 `EngineCoreOutput`，最终只发布一次 finished output，
  然后关闭 Session。

### 12.5 `_patch_scheduler_update_from_output()`

A3 在原生 output update 边界完成以下闭环：

1. 调用 `update_gr_from_output()`，按 `dispatch_id` 回收物理任务并更新逻辑状态；
2. 如果原生 Request 已提前完成、但 Session 仍有后继 Dispatch，则抑制这次终态
   output；
3. 当 `_gr_session_ready_to_finalize()` 成立时，完成原生 Request、退役剩余
   placeholder，并发布 exactly-once 的终态结果；
4. 最终结果沿既有通道返回：

   ```text
   cached BeamBatchResult
       -> EngineCoreOutput.kv_transfer_params["gr_batch_result"]
       -> frontend wait_gr_session()
   ```

这样提前结束的 Prefill 即使已经下发 Decode 后继，也不会丢失最终结果、遗留原生
`RUNNING` Request 或再次调度无 GR metadata 的 token，同时无需新增 result RPC。

### 12.6 `apply_engine_core_child_patches()`

EngineCore 子进程的总安装入口，包括：

- EngineCore request patch；
- Scheduler patch；
- Worker patch；
- KV hash patch。

这保证通过 multiprocessing spawn 创建的新进程也安装 A3 能力。

## 13. `tests/test_gr_async_scheduler.py`：A3 单元测试

### 13.1 辅助函数

`_batch_request()`：

- 构造最小 `BeamBatchRequest`。

`_scheduler()`：

- 构造只包含 A3 所需字段的轻量 Scheduler mock。

`_scheduler_output()`：

- 构造一次原生调度输出。

### 13.2 `test_prefill_decode_lookahead_uses_device_dependency_without_cpu_update()`

验证：

- Prefill 发出后，不等 CPU 消费结果就可以发 Decode；
- Decode 输入引用 Prefill 的设备输出；
- 两个物理 Dispatch in-flight 后禁止继续发；
- 按 `dispatch_id` 回收 Prefill 后重新开放一个调度槽位。

### 13.3 `test_early_finish_stops_issue_and_drains_already_queued_stage()`

验证：

- 前一个 Stage 提前结束后进入 `FINISHING`；
- 不再发新 Stage；
- 已经排队的 Stage 仍然完成物理回收，但不能覆盖终态或再次确认 token；
- 终态结果在 drain 期间被保留；
- 所有 Dispatch drain 后，原生 Request 被完成、placeholder 清零，并且只发布一次
  terminal output。

### 13.4 `test_chunked_prefill_does_not_consume_a_logical_stage()`

验证：

- chunked Prefill metadata 仍然标记为 Prefill；
- `produces_output=False`；
- `issued_stages` 不增加；
- 不消耗 Beam generation budget。

### 13.5 `test_preemption_recompute_does_not_consume_a_decode_stage()`

验证请求已生成一个 token 后被抢占并恢复时：

- 恢复 chunk 仍标记为 `PREFILL`；
- 新下发拥有新的 `dispatch_id`；
- 不推进 `issued_stages` 或 `completed_stages`；
- 不推进 `latest_device_state`，也不消耗剩余 Decode 预算。

### 13.6 `test_backend_configuration_error_uses_request_error_channel()`

验证不支持的 async scheduling 或 batch queue 配置通过
`_handle_request_preproc_error()` 返回当前请求的错误，而不是从提交处理函数抛出并
终止 EngineCore 输入线程。

### 13.7 原生 finish/abort 与终态优先级测试

- `test_native_abort_all_result_cleans_unscheduled_gr_session()`：验证原生
  `finish_requests(None, ...)` 可以清理尚未调度的 GR Session；
- `test_native_abort_overrides_a_pending_gr_terminal_result()`：验证外部 abort 覆盖尚未
  发布的正常终态；
- `test_internal_gr_finalization_preserves_the_pending_terminal_result()`：验证 A3 内部
  完成原生 Request 时不会误删缓存终态。

### 13.8 `test_abort_adapters_treat_randomized_session_id_as_internal()`

验证同步和异步 abort 都使用：

```python
internal=True
```

从而按 A2 生成的内部随机 ID 路由，而不是误用外部 request ID。

## 14. `test_engine_core_protocol_contracts.py`：协议测试

A3 只在已有协议测试中增加新 request type 的检查。

`test_engine_core_wire_value_inventory()` 验证：

```python
SUBMIT_GR_SESSION_WIRE_VALUE == b"\x0a"
```

并检查：

- 所有 EngineCore request type 名字完整；
- 所有 wire value 唯一；
- 重复调用注册函数是幂等的；
- `EngineCoreRequestType(b"\x0a")` 能正确解析为
  `SUBMIT_GR_SESSION`。

## 15. 核心正确性约束

A3 依赖以下不变量：

1. Phase A 的一个 Session 只包含一个原生 Request；
2. `batch_id`、内部 `session_id` 和 EngineCore `request_id` 相同；
3. 每个物理下发拥有唯一、单调递增的 `dispatch_id`，并且最多回收一次；
4. `issued_stages` 和 `completed_stages` 只统计产生输出的逻辑 Stage；
5. 初始 chunked Prefill 和抢占重计算不推进逻辑 Stage 或
   `latest_device_state`；
6. 同一 Session 最多有两个 `_in_flight_dispatches`；
7. 输出型 Stage k 消费 `(session_id, k-1)` 并产生 `(session_id, k)`；
8. A4 必须原样回显 `session_id`、`dispatch_id` 和 `stage_index`；
9. Host completion 只用于回收计数，不定义 GPU 数据依赖；
10. Worker 协议错误必须转换为请求级 failure，不能逃出 EngineCore output 边界；
11. 进入 drain 或外部终态后不再发出新 Stage；
12. 外部 abort/error 优先于尚未发布的正常 terminal result；
13. 仍有物理 Dispatch 未回收时不能删除 Session 或发布最终结果；
14. 清空 placeholder、完成原生 Request、发布 terminal output 和删除 Session 必须形成
    exactly-once 闭环；
15. 中间 Beam Token、Parent 和 Score 不经过 frontend control path。

## 16. 为什么 look-ahead 目前只能是 2

深度 2 针对物理 Dispatch，表示：

```text
当前物理 Dispatch + 一个后继物理 Dispatch
```

继续扩大深度需要后续 Worker 同时提供：

- 多份稳定的 Decode 输入 Buffer；
- 多份 pending Stage 状态；
- 明确的 CUDA Event 所有权；
- KV/Parent 更新的严格依赖；
- 避免后一个 Stage 覆盖前一个 Stage 输入。

在 A4 仍只拥有单份 pending/input 状态时，仅修改
`GR_LOOKAHEAD_DEPTH` 会造成状态覆盖风险，因此 A3 固定为 2。

## 17. A3 的性能收益和仍然存在的开销

A3 消除的是相邻 Stage 之间强制等待 Host Future 的调度依赖。

配合 A4 后，可以把以下 CPU 工作与当前 GPU Stage 重叠：

- Scheduler bookkeeping；
- 下一 Stage metadata 构造；
- Request 状态更新；
- 下一次 ModelExecutor 入队。

A3 不保证消除所有 Decode-to-Decode bubble。以下开销仍属于 A4/Worker：

- Worker `prepare_inputs`；
- Attention metadata 构造；
- CUDA Kernel launch；
- CUDA Graph replay 前的输入准备；
- 最终 D2H 和 Event fence；
- 稳定 Buffer 与双缓冲管理。

## 18. 建议的 A3 评审顺序

1. 检查 `wire.py` 和 `codec.py`，确认新请求协议稳定且无 wire value 冲突；
2. 检查 `core_client.py`、`core_client_patch.py` 和 `core.py`，确认完整
   Session 只提交、预处理一次，后端配置失败走请求错误通道；
3. 检查 `GRSessionState` 的逻辑 Stage 计数、物理 Dispatch 集合和终态所有权；
4. 检查 `attach_gr_stage_metadata()` 是否为每次下发分配唯一 `dispatch_id`，并正确
   构造逻辑 Stage 编号和设备依赖；
5. 检查 `apply_scheduler_patch()` 的物理 look-ahead 限制和 `finally` 恢复；
6. 检查初始 chunked Prefill 和抢占重计算是否都不会消耗逻辑 Stage；
7. 检查 Worker 返回的身份、token 数、设备引用和 terminal result 校验；
8. 检查提前结束后是否缓存终态、停止发新 Stage，并在 drain 后完成原生 Request、
   清空 placeholder、只发布一次结果；
9. 检查 abort-all、外部 abort/error 和内部 GR finalization 的优先级；
10. 检查内部 ID 和外部 request ID 是否严格分离；
11. 最后检查 A4/A5 预留接口是否只传控制信息、不传中间 Beam 数据。

## 19. 建议测试命令

纯 A3 的聚焦测试：

```bash
pytest -q \
  tests/test_gr_async_scheduler.py \
  tests/contract/test_engine_core_protocol_contracts.py \
  tests/test_beam_search_v1_client.py \
  tests/test_beam_search_v1_entrypoint.py \
  tests/test_beam_search_v1_input_processor.py
```

A4-A6 的 Worker、终态输出和端到端 CUDA 测试应在后续集成测试分支执行，
不应将其结果写成纯 A3 本身已经实现的能力。
