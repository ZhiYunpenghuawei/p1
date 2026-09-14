#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Profile the public A5 path with stage and terminal materialization ranges."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from a5_profile_common import (
    DEFAULT_CATALOG,
    DEFAULT_MODEL,
    REPO_ROOT,
    build_prompt,
    check_inputs,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--generated-steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--profiled-requests", type=int, default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--hit", action="store_true")
    parser.add_argument(
        "--with-stack", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _wrap_range(owner: Any, method_name: str, range_name: str, torch: Any) -> bool:
    original = getattr(owner, method_name, None)
    if not callable(original):
        return False

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with torch.autograd.profiler.record_function(range_name):
            return original(*args, **kwargs)

    setattr(owner, method_name, wrapped)
    return True


def _install_ranges(llm: Any, torch: Any) -> tuple[list[str], Any, dict[str, int]]:
    from vllm_gr.v1.worker.beam_final_output import GPUBeamFinalOutput
    from vllm_gr.v1.worker.beam_search_runner import BeamSearchRunner
    from vllm_gr.v1.worker.gpu_beam_stage_runner import (
        AsyncGPUBeamOutput,
        GPUBeamStageRunner,
    )

    core_client = llm.llm_engine.engine_core
    core = getattr(core_client, "engine_core", core_client)
    model_runner = core.model_executor.driver_worker.model_runner
    ranges = (
        (core.scheduler, "schedule", "a5_scheduler_schedule"),
        (core.scheduler, "update_from_output", "a5_scheduler_update"),
        (model_runner, "_prepare_inputs", "a5_worker_prepare_inputs"),
        (model_runner, "execute_model", "a5_worker_execute_model"),
        (model_runner, "sample_tokens", "a5_worker_sample_tokens"),
        (GPUBeamStageRunner, "execute", "a5_stage_execute"),
        (GPUBeamStageRunner, "admit", "a5_stage_admit"),
        (GPUBeamStageRunner, "prepare_decode", "a5_prepare_next_decode"),
        (GPUBeamStageRunner, "_forward", "a5_eager_model_forward"),
        (GPUBeamStageRunner, "sample", "a5_device_beam_decision"),
        (AsyncGPUBeamOutput, "get_output", "a5_async_output_consumer"),
        (BeamSearchRunner, "initialize_from_prefill", "a5_state_initialize"),
        (BeamSearchRunner, "advance", "a5_state_advance"),
        (GPUBeamFinalOutput, "enqueue", "a5_terminal_select_and_d2h_submit"),
        (GPUBeamFinalOutput, "consume", "a5_terminal_result_consume"),
        (GPUBeamStageRunner, "release", "a5_worker_release"),
    )
    installed = [
        name for owner, method, name in ranges if _wrap_range(owner, method, name, torch)
    ]
    stage_runner = getattr(model_runner, "_gr_gpu_stage_runner", None)
    graph_before = {
        "captures": int(getattr(stage_runner, "graph_captures", 0)),
        "replays": int(getattr(stage_runner, "graph_replays", 0)),
    }
    return installed, model_runner, graph_before


def _graph_counts(model_runner: Any, before: dict[str, int]) -> dict[str, int]:
    stage_runner = getattr(model_runner, "_gr_gpu_stage_runner", None)
    if stage_runner is None:
        raise RuntimeError("A5 stage runner was not created while profiling")
    captures = int(getattr(stage_runner, "graph_captures", 0))
    replays = int(getattr(stage_runner, "graph_replays", 0))
    return {
        "captures_before_profile": before["captures"],
        "captures_during_profile": captures - before["captures"],
        "replays_before_profile": before["replays"],
        "replays_during_profile": replays - before["replays"],
    }


def _profiler_activities(torch: Any) -> tuple[list[Any], str]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        return activities, "self_cuda_time_total"
    npu_activity = getattr(torch.profiler.ProfilerActivity, "NPU", None)
    if npu_activity is not None:
        activities.append(npu_activity)
        return activities, "self_npu_time_total"
    return activities, "self_cpu_time_total"


def _validate_outputs(outputs: list[Any], width: int) -> None:
    for output in outputs:
        if len(output) != 1 or len(output[0].sequences) != width:
            raise AssertionError("beam_search_v1 output Beam shape is incorrect")


def _reset_prefix_cache(llm: Any) -> None:
    if not llm.reset_prefix_cache():
        raise RuntimeError("beam_search_v1 prefix-cache reset failed")


def main() -> None:
    args = arguments()
    if args.width <= 0 or args.generated_steps <= 0 or args.profiled_requests <= 0:
        raise ValueError("width, generated-steps and profiled-requests must be positive")
    if args.warmups < 0 or args.prompt_length < 2:
        raise ValueError("warmups and prompt-length are invalid")
    model, catalog = check_inputs(args.model, args.catalog)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("USE_TF", "0")
    if args.graph:
        os.environ.setdefault("VLLM_ENABLE_PREFILL_CUDAGRAPH", "1")

    import torch

    from tools.build_constraint_table import build_constraint_table
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir or Path("trace") / f"a5_profile_{stamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "a5_trace.json"

    table_path = output_dir / "a5_constraint_table.safetensors"
    artifact_digest, tokenizer_digest, _ = build_constraint_table(
        triples_path=catalog,
        model=str(model),
        revision=None,
        output_path=table_path,
        trust_remote_code=True,
    )
    gr_config = {
        "schema_version": 1,
        "beam": {
            "graph_enabled": args.graph,
            "max_width": args.width,
            "max_decode_steps": args.generated_steps,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "cuda",
            "path": str(table_path.resolve()),
            "format": "constraint_table_v1",
            "artifact_digest": artifact_digest,
            "tokenizer_digest": tokenizer_digest,
            "max_top_k": args.width,
        },
        "attention_backend": "CUSTOM",
    }

    activities, sort_key = _profiler_activities(torch)

    with GRLLM(
        model=str(model),
        dtype="bfloat16",
        async_scheduling=True,
        enforce_eager=not args.graph,
        trust_remote_code=True,
        max_logprobs=args.width,
        vllm_gr_config=gr_config,
        max_num_seqs=1,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.3,
        distributed_executor_backend="uni",
        compilation_config=(
            {"mode": 0, "cudagraph_mode": "FULL"} if args.graph else None
        ),
    ) as llm:
        tokenizer = llm.get_tokenizer()
        prompt = build_prompt(tokenizer, args.prompt_length)
        prompts = [{"prompt_token_ids": prompt[:-1]}]
        params = BeamSearchParams(
            beam_width=args.width,
            max_tokens=args.generated_steps + 2,
            temperature=0,
            ignore_eos=True,
            include_stop_str_in_output=True,
            begin_token="<|sid_begin|>",
            end_token="<|sid_end|>",
        )

        for _ in range(args.warmups):
            if not args.hit:
                _reset_prefix_cache(llm)
            llm.beam_search_v1(prompts, params)
        torch.accelerator.synchronize()
        if not args.hit:
            _reset_prefix_cache(llm)
        installed_ranges, model_runner, graph_before = _install_ranges(llm, torch)

        raw_outputs = []
        request_ms = []
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=args.with_stack,
            with_modules=True,
        ) as profiler:
            capture_started = time.perf_counter()
            for index in range(args.profiled_requests):
                if index > 0 and not args.hit:
                    _reset_prefix_cache(llm)
                request_started = time.perf_counter()
                with torch.autograd.profiler.record_function(
                    f"a5_profiled_request_{index}"
                ):
                    raw_outputs.append(llm.beam_search_v1(prompts, params))
                    torch.accelerator.synchronize()
                request_ms.append((time.perf_counter() - request_started) * 1000.0)
            capture_wall_ms = (time.perf_counter() - capture_started) * 1000.0
        _validate_outputs(raw_outputs, args.width)
        graph_counts = _graph_counts(model_runner, graph_before)

    profiler.export_chrome_trace(str(trace_path))
    table = profiler.key_averages(group_by_input_shape=True).table(
        sort_by=sort_key, row_limit=300
    )
    (output_dir / "profiler_table.txt").write_text(table + "\n", encoding="utf-8")
    result = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "profiled_requests": args.profiled_requests,
        "profile_capture_wall_ms": capture_wall_ms,
        "profiled_request_wall_ms": request_ms,
        "timing_note": "Profiler-instrumented wall time; use run_a5_ab.py for latency.",
        "config": {
            "width": args.width,
            "prompt_length": args.prompt_length,
            "generated_steps": args.generated_steps,
            "graph": args.graph,
            "prefix_cache_hit": args.hit,
            "constraint_table_path": str(table_path.resolve()),
            "constraint_table_artifact_digest": artifact_digest,
            "constraint_table_tokenizer_digest": tokenizer_digest,
        },
        "trace": str(trace_path),
        "trace_bytes": trace_path.stat().st_size,
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "graph_counts": graph_counts,
        "ranges": installed_ranges,
    }
    (output_dir / "run.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"A5 profile: {output_dir / 'run.json'}")


if __name__ == "__main__":
    main()
