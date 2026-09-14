#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare legacy and public A5 Beam paths in isolated engine processes."""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from a5_profile_common import (
    DEFAULT_CATALOG,
    DEFAULT_MODEL,
    check_inputs,
    distribution,
    legacy_case_command,
    load_json,
    new_case_command,
    run_child,
    write_json,
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
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--score-atol", type=float, default=1e-4)
    parser.add_argument("--score-rtol", type=float, default=1e-5)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--hit", action="store_true")
    return parser.parse_args()


def _precision(
    legacy: dict[str, Any],
    current: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left_records = legacy["records"]
    right_records = current["records"]
    if len(left_records) != len(right_records):
        raise AssertionError("legacy and A5 measured request counts differ")

    same_rank = 0
    compared = 0
    candidate_overlap = 0
    score_matches = 0
    text_matches = 0
    logprob_shape_matches = 0
    score_diffs: list[float] = []
    legacy_logprob_steps: list[float] = []
    current_logprob_steps: list[float] = []
    first_mismatch_ranks: list[int | None] = []
    for left, right in zip(left_records, right_records, strict=True):
        left_rows = left["sequences"]
        right_rows = right["sequences"]
        if len(left_rows) != len(right_rows):
            raise AssertionError("legacy and A5 Beam counts differ")
        first_mismatch = None
        for rank, (left_row, right_row) in enumerate(
            zip(left_rows, right_rows, strict=True)
        ):
            token_match = left_row["token_ids"] == right_row["token_ids"]
            same_rank += int(token_match)
            compared += 1
            if not token_match and first_mismatch is None:
                first_mismatch = rank
            left_score = float(left_row["score"])
            right_score = float(right_row["score"])
            if math.isfinite(left_score) and math.isfinite(right_score):
                difference = abs(left_score - right_score)
                score_diffs.append(difference)
                score_matches += int(
                    math.isclose(left_score, right_score, abs_tol=atol, rel_tol=rtol)
                )
            text_matches += int(left_row["text_sha256"] == right_row["text_sha256"])
            left_logprobs = int(left_row["logprob_steps"])
            right_logprobs = int(right_row["logprob_steps"])
            legacy_logprob_steps.append(float(left_logprobs))
            current_logprob_steps.append(float(right_logprobs))
            logprob_shape_matches += int(left_logprobs == right_logprobs)
        first_mismatch_ranks.append(first_mismatch)
        left_candidates = {tuple(row["token_ids"]) for row in left_rows}
        right_candidates = {tuple(row["token_ids"]) for row in right_rows}
        candidate_overlap += len(left_candidates & right_candidates)

    return {
        "same_rank_ratio": same_rank / compared if compared else None,
        "candidate_overlap_ratio": candidate_overlap / compared if compared else None,
        "score_match_ratio": score_matches / len(score_diffs) if score_diffs else None,
        "text_exact_match_ratio": text_matches / compared if compared else None,
        "logprob_step_count_match_ratio": (
            logprob_shape_matches / compared if compared else None
        ),
        "legacy_logprob_steps": distribution(legacy_logprob_steps),
        "a5_logprob_steps": distribution(current_logprob_steps),
        "score_abs_diff": distribution(score_diffs),
        "first_mismatch_rank_zero_based": first_mismatch_ranks,
        "score_tolerance": {"atol": atol, "rtol": rtol},
    }


def main() -> None:
    args = arguments()
    if args.width <= 0 or args.generated_steps <= 0 or args.samples <= 0:
        raise ValueError("width, generated-steps and samples must be positive")
    if args.warmups < 0 or args.prompt_length < 2:
        raise ValueError("warmups and prompt-length are invalid")
    model, catalog = check_inputs(args.model, args.catalog)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir or Path("trace") / f"a5_ab_{stamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    legacy_path = output_dir / "legacy.json"
    current_path = output_dir / "a5.json"

    common = dict(
        model=model,
        catalog=catalog,
        width=args.width,
        prompt_length=args.prompt_length,
        steps=args.generated_steps,
        warmups=args.warmups,
        samples=args.samples,
        graph=args.graph,
        hit=args.hit,
    )
    run_child(
        new_case_command(output=current_path, **common),
        output_dir / "a5.log",
    )
    run_child(
        legacy_case_command(output=legacy_path, **common),
        output_dir / "legacy.log",
    )
    legacy = load_json(legacy_path)
    current = load_json(current_path)
    legacy_times = [float(record["request_ms"]) for record in legacy["records"]]
    current_times = [float(record["request_ms"]) for record in current["records"]]
    legacy_materialization = [
        float(record["materialization_ms"]) for record in legacy["records"]
    ]
    current_materialization = [
        float(record["materialization_ms"]) for record in current["records"]
    ]
    legacy_consumed = [
        request + materialization
        for request, materialization in zip(
            legacy_times, legacy_materialization, strict=True
        )
    ]
    current_consumed = [
        request + materialization
        for request, materialization in zip(
            current_times, current_materialization, strict=True
        )
    ]
    legacy_mean = sum(legacy_times) / len(legacy_times)
    current_mean = sum(current_times) / len(current_times)
    report = {
        "config": current["config"],
        "paths": {
            "legacy": legacy["boundary"],
            "a5": current["boundary"],
        },
        "performance_ms": {
            "legacy": distribution(legacy_times),
            "a5": distribution(current_times),
            "legacy_over_a5_mean": legacy_mean / current_mean,
            "lazy_materialization_outside_e2e": {
                "legacy": distribution(legacy_materialization),
                "a5": distribution(current_materialization),
            },
            "request_plus_full_materialization": {
                "legacy": distribution(legacy_consumed),
                "a5": distribution(current_consumed),
            },
        },
        "precision": _precision(
            legacy,
            current,
            atol=args.score_atol,
            rtol=args.score_rtol,
        ),
        "artifacts": {
            "legacy": str(legacy_path),
            "a5": str(current_path),
        },
    }
    write_json(output_dir / "report.json", report)
    print(f"A5 A/B report: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
