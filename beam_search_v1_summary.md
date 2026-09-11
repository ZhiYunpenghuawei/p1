# beam_search_v1 总体说明与使用方式

## 1. 背景与目标

该系列基于上游 `decode_graph`（基线 `91220f4`）继续实现 issue #339。A1 公共调度骨架已经合入上游，因此本系列从 A2 开始。

旧 Beam Search 由主机逐步创建、提交请求并读取中间结果。新的 `beam_search_v1` 将一次 Beam Search 表示为一个长期存在的 session：

```text
一次提交
  -> EngineCore 按阶段调度 Prefill / Decode
  -> Worker 在 GPU 上持续保存 Beam 与 KV 状态
  -> 最终阶段一次性回传结果
  -> release / abort 统一清理
```

主要收益是减少请求重建、CPU/GPU 同步、中间 D2H 传输以及 Beam/KV 状态的重复物化。

## 2. PR 拆分

| 阶段 | 分支 | 主要功能 | Commit 变动 |
|---|---|---|---:|
| A2 | `phase-a-beam-search-v1-a2` | 单次提交、原生请求/session 常驻 | `+497/-23` |
| A3 | `phase-a-beam-search-v1-a3` | EngineCore 单步超前异步调度 | `+694/-1` |
| A4 | `phase-a-beam-search-v1-a4` | GPU Beam 状态、约束状态与 KV 重排常驻 | `+1769/-7` |
| A5 | `phase-a-beam-search-v1-a5` | 仅最终输出及统一清理 | `+465/-0` |
| A6 | `phase-a-beam-search-v1-a6` | ModelRunner、客户端与入口端到端集成 | `+440/-33` |

五个 commit 逐个统计合计为 `+3865/-64`；相对上游基线的最终净差异为 32 个文件、`+3833/-32`。两组数字不同，是因为后续 commit 会继续修改前面 commit 新增的代码。

依赖顺序为：

```text
decode_graph -> A2 -> A3 -> A4 -> A5 -> A6
```

## 3. 新调用方式

公开的离线入口从原来的：

```python
outputs = llm.beam_search(prompts, params)
```

改为显式调用：

```python
outputs = llm.beam_search_v1(prompts, params)
```

完整示例：

```python
from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams

beam_width = 128
decode_steps = 3

with GRLLM(
    model="/path/to/model",
    catalog_path="/path/to/video_constraint_triples.json",
    constraint_backend="constraint_table",
    trust_remote_code=True,
    max_logprobs=beam_width,
    beam_graph_enabled=True,
    beam_max_width=beam_width,
    beam_max_decode_steps=decode_steps,
    attention_config={"backend": "CUSTOM"},
    enable_prefix_caching=True,
    enable_chunked_prefill=False,
    async_scheduling=True,
    max_model_len=8192,
    max_num_batched_tokens=8192,
    max_num_seqs=1,
    enforce_eager=False,
    compilation_config={"mode": 0, "cudagraph_mode": "FULL"},
) as llm:
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = tokenizer.encode("your prompt")

    params = BeamSearchParams(
        beam_width=beam_width,
        # begin/end token 各占一个 API max_tokens；实际 Decode 为 3 步。
        max_tokens=decode_steps + 2,
        temperature=0.0,
        ignore_eos=True,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
        include_stop_str_in_output=True,
        length_penalty=1.0,
    )

    outputs = llm.beam_search_v1(
        [{"prompt_token_ids": prompt_token_ids}],
        params,
    )

    for sequence in outputs[0].sequences:
        print(sequence.text, sequence.cum_logprob)
```

如果模型不使用 `begin_token`/`end_token`，则 `max_tokens` 直接填写实际 Decode 步数。

## 4. 当前约束

- 仅支持 CUDA。
- 当前要求 `TP=PP=DP=1`。
- 每次调用只支持一个 prompt，且 `max_num_seqs=1`。
- `BeamSearchParams.beam_width` 必须等于启动时的 `beam_max_width`。
- 当前标准 SID 约束最多支持 3 个实际 Decode stage。
- 必须使用固定步数：`ignore_eos=True`，不能启用 early stopping。
- 当前精度基线要求 `compilation_config.mode=0`。
- 暂不支持 LoRA、多模态、embedding prompt、speculative 或 hybrid model。
- prompt 长度及 Beam width 都不能超过 `max_num_batched_tokens` 预算。

## 5. 在线调用说明

代码同时提供了 `OpenAIServing.beam_search_v1(...)` 异步内部入口，它只 yield 最终结果。但当前系列没有新增独立 REST 路由；对外最直接、完整的使用方式仍是 `GRLLM.beam_search_v1(...)`。若业务服务需要通过 HTTP 选择 V1，还需在对应请求 handler 中显式调用该异步方法。

