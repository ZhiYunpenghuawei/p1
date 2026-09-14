# SPDX-License-Identifier: Apache-2.0

"""Run one isolated legacy constrained Beam case with normalized final rows."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from a5_profile_common import (
    DEFAULT_CATALOG,
    DEFAULT_MODEL,
    REPO_ROOT,
    build_prompt,
    materialize_beam_output,
    write_json,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--generated-steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--hit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.width <= 0 or args.generated_steps <= 0 or args.samples <= 0:
        raise ValueError("width, generated-steps and samples must be positive")
    if args.warmups < 0 or args.prompt_length < 2:
        raise ValueError("warmups and prompt-length are invalid")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch

    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    records: list[dict[str, Any]] = []
    with GRLLM(
        model=str(args.model.resolve()),
        dtype="bfloat16",
        async_scheduling=True,
        enforce_eager=not args.graph,
        trust_remote_code=True,
        max_logprobs=args.width,
        catalog_path=str(args.catalog.resolve()),
        constraint_backend="constraint_table",
        beam_max_width=args.width,
        beam_max_decode_steps=args.generated_steps,
        beam_graph_enabled=args.graph,
        attention_config={"backend": "CUSTOM"},
        max_num_seqs=1,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.3,
        distributed_executor_backend="uni",
        additional_config={
            "vllm_gr_beam_worker_decision": False,
            "vllm_gr_beam_engine_driven": False,
        },
    ) as llm:
        tokenizer = llm.get_tokenizer()
        prompt = build_prompt(tokenizer, args.prompt_length)
        params = BeamSearchParams(
            beam_width=args.width,
            max_tokens=args.generated_steps + 2,
            temperature=0,
            ignore_eos=True,
            include_stop_str_in_output=True,
            begin_token="<|sid_begin|>",
            end_token="<|sid_end|>",
        )

        def run() -> dict[str, Any]:
            if not args.hit and not llm.reset_prefix_cache():
                raise RuntimeError("legacy prefix-cache reset failed")
            begin = time.perf_counter()
            output = llm.beam_search([{"prompt_token_ids": prompt[:-1]}], params)
            torch.accelerator.synchronize()
            elapsed = (time.perf_counter() - begin) * 1000.0
            materialization_ms, sequences = materialize_beam_output(
                output, args.width, args.prompt_length
            )
            return {
                "request_ms": elapsed,
                "materialization_ms": materialization_ms,
                "sequences": sequences,
            }

        for _ in range(args.warmups):
            run()
        for _ in range(args.samples):
            records.append(run())

    write_json(
        args.output,
        {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "boundary": (
                "public beam_search with frontend beam decision and "
                "engine-driven chaining disabled; full Beam field materialization "
                "is measured separately"
            ),
            "config": {
                "width": args.width,
                "prompt_length": args.prompt_length,
                "generated_steps": args.generated_steps,
                "graph": args.graph,
                "prefix_cache_hit": args.hit,
                "beam_worker_decision": False,
                "beam_engine_driven": False,
            },
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
