# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# Direct script execution puts only ``tests/profiling`` on ``sys.path``.
# Prefer the worktree owning these scripts over any editable installation.
repo_path = str(REPO_ROOT)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
DEFAULT_MODEL = Path("/home/z00980960/OneRec/OneRec-1.7B")
DEFAULT_CATALOG = REPO_ROOT / "tests/resources/generated_data/video_constraint_triples.json"
DEFAULT_PROMPT = REPO_ROOT / "tests/resources/single_one_rec_prompt.txt"


def build_prompt(tokenizer: Any, length: int) -> list[int]:
    """Apply the legacy left-padding policy at an exact prepared length."""
    source = tokenizer.encode(
        DEFAULT_PROMPT.read_text(encoding="utf-8"), add_special_tokens=False
    )
    target = length - 1
    padding = tokenizer.encode("\n", add_special_tokens=False)
    if len(padding) != 1:
        raise RuntimeError(f"newline must map to one token, got {padding}")
    prompt = ([padding[0]] * max(0, target - len(source)) + source)[-target:]
    begin = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    if begin in (None, -1):
        raise RuntimeError("tokenizer is missing <|sid_begin|>")
    prompt.append(begin)
    if len(prompt) != length:
        raise AssertionError("prepared prompt length does not match the requested length")
    return [int(token) for token in prompt]


def check_inputs(model: Path, catalog: Path) -> tuple[Path, Path]:
    model = model.expanduser().resolve()
    catalog = catalog.expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"model directory not found: {model}")
    if not catalog.is_file():
        raise FileNotFoundError(f"catalog file not found: {catalog}")
    return model, catalog


def run_child(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"child process failed; inspect {log_path}", file=sys.stderr)
            raise


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
        "max": ordered[-1],
    }


def materialize_beam_output(
    output: list[Any], width: int, prompt_length: int
) -> tuple[float, list[dict[str, Any]]]:
    """Force lazy public fields outside E2E timing and normalize for comparison."""

    if len(output) != 1 or len(output[0].sequences) != width:
        raise AssertionError("Beam output shape is incorrect")
    started = time.perf_counter()
    for sequence in output[0].sequences:
        _ = sequence.tokens
        _ = sequence.text
        _ = sequence.logprobs
    materialization_ms = (time.perf_counter() - started) * 1000.0

    rows = []
    for sequence in output[0].sequences:
        tokens = [int(token) for token in sequence.tokens]
        text = sequence.text or ""
        rows.append(
            {
                "token_ids": tokens[prompt_length:-1],
                "score": float(sequence.cum_logprob),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text_length": len(text),
                "logprob_steps": len(sequence.logprobs),
            }
        )
    return materialization_ms, rows


def new_case_command(
    *,
    model: Path,
    catalog: Path,
    output: Path,
    width: int,
    prompt_length: int,
    steps: int,
    warmups: int,
    samples: int,
    graph: bool,
    hit: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("a5_new_case.py")),
        "--model",
        str(model),
        "--catalog",
        str(catalog),
        "--output",
        str(output),
        "--width",
        str(width),
        "--prompt-length",
        str(prompt_length),
        "--generated-steps",
        str(steps),
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
    ]
    if graph:
        command.append("--graph")
    if hit:
        command.append("--hit")
    return command


def legacy_case_command(
    *,
    model: Path,
    catalog: Path,
    output: Path,
    width: int,
    prompt_length: int,
    steps: int,
    warmups: int,
    samples: int,
    graph: bool,
    hit: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("a5_legacy_case.py")),
        "--model",
        str(model),
        "--catalog",
        str(catalog),
        "--output",
        str(output),
        "--width",
        str(width),
        "--prompt-length",
        str(prompt_length),
        "--generated-steps",
        str(steps),
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
    ]
    if graph:
        command.append("--graph")
    if hit:
        command.append("--hit")
    return command
