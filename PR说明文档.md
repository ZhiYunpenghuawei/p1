RECIF Model and Shared Beam Search Design
1. 文档目的
本文说明当前 PR 中两部分核心改动：

在 vllm-gr 中支持 RECIF 模型的加载、离线 beam search 和在线服务。
整理 GRLLM.beam_search()，让普通模型和具有分层搜索策略的模型复用公共搜索主干，并使 CUSTOM 后端支持 RECIF。
当前设计遵循以下边界：

vllm-gr 负责模型执行、候选计算、全局 beam 排序、剪枝和 KV Cache 优化。
RECIF 的层数、每层分支数和每层 token 范围属于模型搜索语义。
SID 与 token ID 的相互转换属于调用方，不属于通用推理框架。
原始训练权重到 Hugging Face 权重的转换是离线部署操作，不在模型加载时执行。
2. 总体设计
原始 RECIF checkpoint
  ├── config.json
  ├── _model_rank0.pt
  └── external_rank0.pt
            │
            │ tools/convert_recif_checkpoint.py
            ▼
标准运行时 checkpoint
  ├── config.json             architectures = ["RecifForCausalLM"]
  └── model.safetensors       Qwen3-MoE backbone + 合并后的 lm_head
            │
            ▼
vLLM general plugin
  ├── 注册 RecifForCausalLM
  └── 注册 RECIF BeamSearchStep builder
            │
      ┌─────┴─────────────┐
      ▼                   ▼
离线 GRLLM.beam_search   在线 /v1/completions
      │                   │
      └───────┬───────────┘
              ▼
      RECIF 分层搜索策略
      expand: 8 / 8 / 8
      keep:   8 / 64 / 64（beam_width=64 时）
      range:  level 0 / level 1 / level 2
              │
              ▼
      原始生成 token IDs
              │
              │ 调用方 recif_utils
              ▼
             SID
RECIF 与普通 OneRec 搜索的公共部分是：

使用 BeamSearchStep 表示单层策略；
每个父 beam 扩展候选；
候选分数加上父 beam 累积分数；
在所有父 beam 的候选之间做全局 Top-K；
创建下一层 beam 并记录父子关系；
CUSTOM 后端复用 grouped beam 和 KV Cache。
主要区别是约束时机：

RECIF:
allowed_token_ids → vLLM 屏蔽其他 token → 返回 top candidates → 全局剪枝

OneRec Catalog:
返回 top candidates → catalog.valid() 按父 beam 过滤 → 全局剪枝
3. RECIF 配置的四级归属
配置按照生命周期分为四类。这样可以避免把模型结构、引擎容量和单次请求混在同一个参数对象中。

3.1 Checkpoint / Hugging Face 模型配置
这类参数描述神经网络本身，写在转换后 checkpoint 的 config.json 中，由 vLLM 加载模型时读取。

参数	当前含义	传入时机
architectures	设置为 RecifForCausalLM，用于 vLLM ModelRegistry 选择模型类	checkpoint 转换时写入
vocab_size	RECIF 输出总词表大小，要求为 3 × 8192 = 24576	模型加载时读取并校验
tie_word_embeddings	必须为 false，因为三个 SID head 不与输入 embedding 共享	模型加载时读取并校验
hidden_size	Qwen3-MoE 隐藏层宽度，也是 SID head 的输入宽度	权重转换和模型加载时读取
num_hidden_layers	Transformer 层数	权重转换和模型加载时读取
num_attention_heads	Query attention head 数量	权重转换和模型加载时读取
num_key_value_heads	KV head 数量	权重转换和模型加载时读取
head_dim	单个 attention head 的维度	权重转换和模型加载时读取
num_experts	MoE expert 数量	权重转换和模型加载时读取
其他 Qwen3-MoE 参数	RoPE、激活函数、精度等标准模型配置	由 vLLM/Qwen3-MoE 实现读取
这些参数不应在每次 beam-search 请求中改变。RecifForCausalLM 继承 vLLM 的 Qwen3MoeForCausalLM，只增加 RECIF 必需的配置校验，不在运行时执行权重格式转换。

3.2 RECIF 模型内在搜索策略
这类参数描述 RECIF 的分层 SID 空间，目前内化在 vllm_gr/models/recif.py。

参数	当前值	含义
VOCAB	8192	每一级 SID 的类别数
NUM_SID_LEVELS	3	RECIF 固定生成三个 SID level
FULL_VOCAB	24576	三个互不重叠 level 的总词表大小
_RECIF_BRANCH_FACTORS	(8, 8, 8)	每个父 beam 在每一层扩展8个候选
_RECIF_ALLOWED_TOKEN_RANGES	[0,8192)、[8192,16384)、[16384,24576)	每层允许参与当前层排序的 token
RECIF_MAX_LOGPROBS	8	当前模型策略所需的最小 max_logprobs 容量
build_beam_search_steps() 将这些配置转换成公共结构：

BeamSearchStep(
    expand_width=8,
    keep_width=min(当前累计候选数, beam_width),
    allowed_token_ids=当前层token范围,
)
当 beam_width=64 时，三个 step 为：

层	父 beam 数	每个父 beam 扩展	候选上限	最终保留
Level 0	1	8	8	8
Level 1	8	8	64	64
Level 2	64	8	512	64
branch_factors 在 build_beam_search_steps() 中保留了一个 keyword-only 接口，供未来接入启动级模型配置。目前没有暴露为 GRLLM 参数、命令行参数或 request 参数，生产路径始终使用 (8, 8, 8)。

3.3 引擎启动配置
这类参数在创建 GRLLM 或启动 API Server 时传入，控制模型实例、引擎能力和执行后端。同一模型实例生命周期内保持不变。

参数	RECIF 推荐值	含义
model	转换后的 checkpoint_hf	标准 HF checkpoint 路径
skip_tokenizer_init	True	调用方直接传 prompt_token_ids，运行时不加载 tokenizer
max_logprobs	至少 8	引擎允许单步返回的最大候选数，必须不小于最大 expand_width
max_num_seqs	通常不小于目标 beam 数，例如 64	调度器并发序列容量，不代表搜索分支数
logprobs_mode	processed_logprobs	返回应用 allowed_token_ids 等处理后的分数
dtype	例如 bfloat16	vLLM 标准模型计算精度
tensor_parallel_size	按硬件配置	vLLM 标准张量并行规模
attention_config.backend	默认后端或 CUSTOM	是否启用 vllm-gr grouped beam/KV Cache 优化
需要特别区分：

max_logprobs=8 是引擎容量上限。
_RECIF_BRANCH_FACTORS=(8,8,8) 是模型搜索策略。
max_num_seqs=64 是调度容量。
它们数值可能相同或相关，但语义不同，不能互相替代。
当前没有新增 RECIF 专用的公共启动参数。未来如需配置 branch factor，可以在启动阶段把配置传入预留接口，但不应允许单次 request 修改模型层次结构。

3.4 单次 beam-search 请求配置
这类参数可以在同一个模型实例的不同请求之间变化。

参数	示例	含义
beam_width / 在线 n	64	最终最多保留或返回多少条序列
max_tokens	3	最多生成多少层；RECIF 要求等于 NUM_SID_LEVELS
temperature	0.0	当前 beam search 的分数生成温度
ignore_eos	True	RECIF 固定生成三层，不因普通 EOS 提前终止
return_token_ids	在线请求为 true	让 /v1/completions 返回 choices[].token_ids
stream	当前必须为 false	RECIF 在线 beam search 当前不支持流式响应
请求级 beam_width 不改变 RECIF 每个父节点扩展8个候选的结构。例如 beam_width=20 时仍然逐层扩展8个，但每层 keep_width 依次为 8、20、20。

3.5 应用层输入输出语义
SID 转换不是模型参数，也不属于 vllm-gr 运行时。

examples/offline_inference/beam_search/recif_utils.py 提供示例实现：

函数	作用
sid_to_levels()	将一个 SID 拆成三个 level 值
levels_to_sid()	将三个 level 值恢复成 SID
build_prefix()	将历史 SID 转成标准 prompt_token_ids
generated_tokens_to_sid()	将模型生成的三个 token 转成 SID
真实业务可以复用该文件，也可以提供自己的 tokenizer/codec。框架的公共输出只依赖生成 token 和累计分数，不在 RecifForCausalLM 或 beam-search 主干中转换 SID。

4. 模型与策略注册
vllm-gr 通过 general plugin 在每个相关进程启动时执行 initialize_runtime()：

vLLM load_general_plugins()
  → vllm_gr.plugin.register()
  → initialize_runtime()
      ├── 安装 vllm-gr runtime patches
      └── register_recif_model()
          ├── ModelRegistry.register_model("RecifForCausalLM", ...)
          └── register_beam_search_step_builder(
                  "RecifForCausalLM",
                  build_beam_search_steps,
              )
公共 beam-search 入口不接收 beam_search_model="recif"，也不创建 ModelAdapter。它从：

self.llm_engine.model_config.architectures
获取当前架构，并在 _MODEL_STEP_BUILDERS 中查询模型注册的策略。

如果找到 RecifForCausalLM，调用 RECIF builder；否则使用普通固定宽度策略：

BeamSearchStep(
    expand_width=beam_width,
    keep_width=beam_width,
)
这样 OneRec 不需要专用 adapter，也不会进入 RECIF 的层级 token 范围。

5. 离线 beam search 调用流程
5.1 外部调用链
history_sids
  → recif_utils.build_prefix()
  → {"prompt_token_ids": [...]}
  → GRLLM.beam_search(prompts, params)
  → BeamSearchOutput
  → sequence.tokens
  → recif_utils.generated_tokens_to_sid()
调用方传给框架的是标准 vLLM token prompt：

outputs = llm.beam_search(
    [{"prompt_token_ids": prompt_token_ids}],
    params,
)
5.2 GRLLM.beam_search() 内部调用链
GRLLM.beam_search()
  ├── _preprocess_cmpl(prompts)
  │     标准化 TextPrompt / TokensPrompt
  │
  ├── build_beam_search_steps(architectures, max_tokens, beam_width)
  │     ├── RECIF → recif.build_beam_search_steps()
  │     └── 其他模型 → 固定宽度 BeamSearchStep
  │
  ├── 构造 BeamSearchInstance
  │
  ├── 判断 attention backend
  │     ├── 普通后端 → _run_beam_search_steps()
  │     └── CUSTOM   → _custom_beam_search_batch()
  │
  └── select_best_beams()
        → BeamSearchOutput
5.3 普通后端主干
_run_beam_search_steps() 对每个 BeamSearchStep 执行：

当前所有 active beams
  → _build_step_sampling_params(step)
      logprobs = step.expand_width
      allowed_token_ids = step.allowed_token_ids
  → llm.generate(max_tokens=1)
  → apply_beam_search_step()
      ├── 读取各父 beam 的候选
      ├── 累加 parent.cum_logprob
      ├── 可选 Catalog 动态过滤
      ├── 全局 Top-K(step.keep_width)
      └── 生成下一层 beams
RECIF 的 allowed_token_ids 通过 vLLM 原生 SamplingParams 进入 logits processor。在返回 top candidates 之前，其他 level 的 token 已被屏蔽。

5.4 CUSTOM 后端主干
_custom_beam_search_batch() 保持相同的逐层数学逻辑，但改变执行组织方式：

第0层
  → ADD_BATCH 提交初始请求并完成共享 prompt prefill
  → 缓存 session/prefix/KV 状态

后续层
  → 根据上层 fork_info 构造 BeamRequestStepUpdate
  → 一条 grouped message 描述所有 parent/child beams
  → EngineCore 复用共享 prefix 和父 beam KV Cache
  → 返回扁平化的多 beam logprobs
  → _parse_step_logprobs() 恢复父 beam 归属并累积分数
  → select_top_indices(step.keep_width)
  → materialize_selected_beams()
  → 得到 new_beams 和下一层 fork_info
CUSTOM 的主要优化包括：

多个 beam 作为一个逻辑请求提交，减少请求、通信和 scheduler 管理开销；
显式携带 parent/child 关系，复用共享 prefix 与 KV Cache；
使用 step.expand_width，RECIF 每个父 beam 只返回8个候选，而不是返回全局 beam_width 个；
使用 step.keep_width，支持 8 → 64 → 64 的逐层保留宽度；
不依赖 tokenizer，只传可选的 eos_token_id；
普通与 CUSTOM 后端共用 _build_step_sampling_params()，保证 allowed IDs 和扩展宽度一致。
6. 在线调用流程
RECIF 复用标准 POST /v1/completions 路由，没有新增 /v1/recif/... 端点。

POST /v1/completions
  → OpenAIServingCompletion._create_completion()
  → RECIF completion handler patch
      ├── 通过 model_config.architectures 判断 RecifForCausalLM
      ├── 校验 use_beam_search / max_tokens / return_token_ids
      └── recif_beam_search()
          ├── build_beam_search_steps(3, request.n)
          ├── 每层 _add_batch_step(max_tokens=1)
          ├── apply_beam_search_step()
          └── RequestOutput(text="", token_ids=生成的三个token)
  → vLLM CompletionResponse
      └── choices[].token_ids
  → 客户端 generated_tokens_to_sid()
在线和离线的执行入口不同：

离线入口是同步 GRLLM.beam_search()。
在线入口基于 vLLM async engine，由 recif_beam_search() 编排。
但两者复用以下核心语义：

同一个 RECIF build_beam_search_steps()；
同一个 BeamSearchStep；
同一个 apply_beam_search_step() 候选累计和剪枝逻辑；
都只向调用方返回 token IDs，不在框架内部转 SID。
当前在线限制：

只接受一个 token-ID prompt；
use_beam_search=true；
max_tokens=3；
stream=false；
echo=false；
return_token_ids=true。
7. 权重转换与运行时加载
7.1 为什么独立转换
原始 RECIF checkpoint 包含：

_model_rank0.pt   Megatron 格式 Qwen3-MoE backbone
external_rank0.pt 三个独立 SID heads
权重名称转换和 head 合并与推理调度无关，不应在每次模型启动时执行。因此转换逻辑放在：

tools/convert_recif_checkpoint.py
tools/ 不属于 vllm_gr Python package，运行时模型不会导入该脚本。

7.2 转换内容
转换脚本执行：

校验 config.json、_model_rank0.pt、external_rank0.pt。

根据 config 读取 Qwen3-MoE 层数、head、hidden size 和 expert 数量。

将 Megatron QKV、MLP 和 expert 权重名称转换为 HF Qwen3-MoE 名称。

将三个 [8192, hidden_size] SID head 按 level 顺序拼接为：

lm_head.weight: [24576, hidden_size]
保存 model.safetensors。

将 config.json 的 architectures 写为 RecifForCausalLM。

运行时 RecifForCausalLM 直接继承标准 Qwen3-MoE load_weights()。

8. 主要代码结构
vllm_gr/
├── entrypoints/
│   ├── gr.py
│   │   ├── GRLLM.beam_search
│   │   ├── _run_beam_search_steps
│   │   ├── _custom_beam_search_batch
│   │   └── _build_step_sampling_params
│   ├── beam_search_config.py
│   │   ├── BeamSearchStep
│   │   ├── register_beam_search_step_builder
│   │   └── build_beam_search_steps
│   └── recif/
│       ├── completion_handler.py
│       └── serving_engine.py
├── models/
│   └── recif.py
│       ├── RecifForCausalLM
│       ├── build_beam_search_steps
│       └── register_recif_model
├── beam_search_step.py
│   └── apply_beam_search_step
└── plugin.py

examples/offline_inference/beam_search/
├── offline_recif_beam_search.py
└── recif_utils.py

tools/
└── convert_recif_checkpoint.py

tests/
├── test_recif_model_support.py
├── test_shared_beam_step.py
└── system/
    ├── run_recif_e2e.sh
    └── verify_recif_e2e.py
已经删除的旧抽象包括：

ModelAdapter；
OneRecModelAdapter；
RecifModelAdapter；
beam_search_model=BeamSearchModel.RECIF 显式选择；
GRLLM.beam_search() 中的 history_sids 特殊输入；
RECIF 模型加载过程中的动态权重转换；
RECIF 框架侧的 SID 编码和解码。
9. 本地测试方法
下面命令默认在仓库根目录执行。

9.1 安装开发和 RECIF 依赖
uv venv --seed .venv
source .venv/bin/activate
uv pip install vllm==0.22.1
VLLM_GR_TARGET_DEVICE=cuda uv pip install -e ".[dev]"
uv pip install -r requirements/recif.txt
NPU 环境将 VLLM_GR_TARGET_DEVICE 设置为对应平台值。

9.2 代码风格和单元测试
pre-commit run --all-files

pytest \
  tests/test_recif_model_support.py \
  tests/test_shared_beam_step.py
单元测试覆盖：

SID 编解码和 prefix 构造示例；
RECIF 三层 allowed-token 范围；
8/8/8 扩展和 8/64/64 保留策略；
启动级 branch-factor 预留接口；
普通模型固定宽度回归；
RECIF 模型/在线 handler 识别；
三个 SID head 合并和错误处理；
共享 beam step 的累计分数、父 beam 归属和全局 Top-K；
SamplingParams 正确携带 expand_width 和 allowed_token_ids。
9.3 转换本地 checkpoint
原始模型假设位于 ../checkpoint：

python tools/convert_recif_checkpoint.py \
  --input ../checkpoint \
  --output ../checkpoint_hf
转换完成后至少应存在：

../checkpoint_hf/config.json
../checkpoint_hf/model.safetensors
9.4 离线模型测试
python ./examples/offline_inference/beam_search/offline_recif_beam_search.py \
  --model_path ../checkpoint_hf \
  --history 598080194427,628177754964,755993681678 \
  --beam 64
输出格式：

rank sid cumulative_logprob
例如：

0 755993681678 -2.9582809917628765
1 1574911709347 -4.561697155237198
2 741270118158 -4.69528728723526
这里的 SID 是离线示例在框架返回 token 后通过 recif_utils 转换得到的，不是 GRLLM.beam_search() 内部生成的文本。

9.5 启动在线服务
vllm-gr serve ../checkpoint_hf \
  --host 0.0.0.0 \
  --port 8000 \
  --skip-tokenizer-init \
  --max-logprobs 8 \
  --max-num-seqs 64 \
  --logprobs-mode processed_logprobs
9.6 在线请求
调用方先通过 recif_utils.build_prefix() 生成 prompt_token_ids，然后请求标准 completions API：

curl -sS -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": [0,1915,8558,18612,0,6996,10559,18724,0,7950,13040,19200,0],
    "use_beam_search": true,
    "n": 64,
    "max_tokens": 3,
    "return_token_ids": true
  }'
在线响应中的：

{
  "text": "",
  "token_ids": [1915, 8558, 18612]
}
由客户端调用：

generated_tokens_to_sid(choice["token_ids"])
转换为最终 SID。

9.7 完整 RECIF E2E
E2E 脚本接收原始 checkpoint 路径，在临时目录中自动转换权重，然后依次执行离线和在线验证：

RECIF_MODEL_PATH=/path/to/original/checkpoint \
  bash ./tests/system/run_recif_e2e.sh
脚本执行顺序：

校验原始三个文件
  → 转换为临时 HF checkpoint
  → 运行离线 beam search
  → 验证离线64条结果
  → 启动在线服务
  → 请求 /v1/completions
  → 从 choices[].token_ids 转换 SID
  → 验证在线64条结果
verify_recif_e2e.py 对前20个 SID 做集合和近似排名验证：允许少量相邻次序变化，但不允许主要结果集合发生变化。

10. CI 接入
当前 GitHub Actions 使用 self-hosted runner，并从机器本地路径读取未公开的 RECIF checkpoint：

/data/datasets/recif_beam_search/checkpoint
workflow 中执行：

pytest tests/test_recif_model_support.py tests/test_shared_beam_step.py

RECIF_MODEL_PATH=/data/datasets/recif_beam_search/checkpoint \
  bash ./tests/system/run_recif_e2e.sh
模型不通过 GitHub Actions 网络下载，也不提交到仓库。权重只需要存在于执行该 job 的 self-hosted runner 上。

11. 当前设计结论
当前实现最终形成了以下职责边界：

组件	负责内容	不负责内容
RecifForCausalLM	Qwen3-MoE 执行类、模型配置校验	SID 编解码、运行时权重转换
RECIF step builder	三层结构、8/8/8、allowed-token 范围	最终请求 beam width
GRLLM.beam_search()	标准输入、实例管理、公共搜索编排	判断 history_sids、转换 SID
apply_beam_search_step()	累计分数、全局剪枝、构造新 beam	模型专用输入语义
CUSTOM backend	grouped request、fork、KV Cache 和 attention 优化	改变 beam-search 数学结果
recif_utils.py	SID 与 token IDs 转换	模型执行和调度
checkpoint converter	部署前权重格式转换	在线/离线请求处理
因此，vllm-gr 的公共接口保持接近 vLLM：输入是文本或 token IDs，输出核心是 token IDs 和分数；RECIF 的模型结构由模型文件注册，业务 SID 语义由调用方处理。
