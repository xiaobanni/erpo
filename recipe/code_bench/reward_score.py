"""
Code-benchmark scoring helpers shared by the RL reward manager and offline evaluation.

Scores generated code against per-question benchmark JSON files produced by the
`prepare_*` scripts in this directory (LiveCodeBench v6, CodeContests,
CodeForces, TACO). Code is executed in a separate process with a wall-clock and
memory watchdog (`check_correctness`).

Group-aware RL rewards (public_binary / fmv_pointwise) are computed by
`recipe.code_bench.test_rl_reward_manager.CodeBenchTestRLRewardManager`.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from recipe.code_bench.livecodebench import (
    build_in_outs,
    extract_code_from_response,
    load_codegen_problem,
)
from recipe.code_bench.testing_util import run_test

PROCESS_POLL_INTERVAL_SECONDS = 0.1
DEFAULT_MAX_MEMORY_GB = 50
REWARD_MODE_PUBLIC_BINARY = "public_binary"
REWARD_MODE_FMV_POINTWISE = "fmv_pointwise"

_REWARD_MODE_ALIASES = {
    REWARD_MODE_PUBLIC_BINARY: REWARD_MODE_PUBLIC_BINARY,
    "public_pass_fail": REWARD_MODE_PUBLIC_BINARY,
    # PCR reward: scored group-wise by CodeBenchTestRLRewardManager
    # (behavioral consensus on pooled public + probe inputs).
    REWARD_MODE_FMV_POINTWISE: REWARD_MODE_FMV_POINTWISE,
    "fmv": REWARD_MODE_FMV_POINTWISE,
    "fmv-pointwise": REWARD_MODE_FMV_POINTWISE,
}


def _extract_question_id(ground_truth: Any) -> str:
    if not isinstance(ground_truth, dict):
        raise TypeError(f"ground_truth must be a dict, got {type(ground_truth)}")

    question_id = ground_truth.get("question_id")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError(f"ground_truth must contain a non-empty string question_id, got {question_id!r}")
    return question_id


@lru_cache(maxsize=None)
def load_benchmark(question_id: str, benchmark_dir: str) -> Any:
    benchmark_path = Path(benchmark_dir) / f"{question_id}.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Missing benchmark JSON for question_id={question_id}: {benchmark_path}")
    return load_codegen_problem(benchmark_path)


def _normalize_reward_mode(raw_value: str) -> str:
    normalized = _REWARD_MODE_ALIASES.get(raw_value)
    if normalized is None:
        raise ValueError(
            f"Unsupported reward_mode={raw_value!r}. Expected one of {sorted(_REWARD_MODE_ALIASES)}"
        )
    return normalized


def _run_test_worker(
    in_outs: dict[str, Any],
    code: str,
    timeout: int,
    output: Any,
    complete_evaluation: bool,
) -> None:
    sample = {"input_output": json.dumps(in_outs, ensure_ascii=False)}
    test_result, metadata = run_test(
        sample,
        test=code,
        debug=False,
        timeout=timeout,
        complete_evaluation=complete_evaluation,
    )
    output.append((test_result, metadata))


def _aggregate_rss_bytes(pid: int) -> int:
    process = psutil.Process(pid)
    total_memory = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total_memory += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total_memory


def _process_failed_result(
    in_outs: dict[str, Any],
    *,
    error_message: str,
    error_code: int,
    exitcode: int | None,
    elapsed_seconds: float,
    observed_memory_bytes: int | None = None,
    memory_limit_bytes: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    return [-1 for _ in in_outs["inputs"]], {
        "error": error_message,
        "error_code": error_code,
        "error_message": error_message,
        "exitcode": exitcode,
        "elapsed_seconds": elapsed_seconds,
        "num_tests": len(in_outs["inputs"]),
        "observed_memory_bytes": observed_memory_bytes,
        "memory_limit_bytes": memory_limit_bytes,
    }


def check_correctness(
    in_outs: dict[str, Any],
    code: str,
    timeout: int,
    max_memory_gb: int = DEFAULT_MAX_MEMORY_GB,
    complete_evaluation: bool = False,
) -> tuple[list[Any], Any]:
    manager = multiprocessing.Manager()
    output = manager.list()
    process = multiprocessing.Process(
        target=_run_test_worker,
        args=(in_outs, code, timeout, output, complete_evaluation),
    )
    process.start()

    start_time = time.time()
    timeout_seconds = max((timeout + 1) * len(in_outs["inputs"]) + 5, 300)
    memory_limit_bytes = max_memory_gb * 1024 * 1024 * 1024
    observed_memory_bytes: int | None = None
    timed_out = False
    memory_limit_exceeded = False

    try:
        while process.is_alive():
            try:
                pid = process.pid
                if pid is None:
                    break
                observed_memory_bytes = _aggregate_rss_bytes(pid)
            except psutil.NoSuchProcess:
                break

            if observed_memory_bytes > memory_limit_bytes:
                memory_limit_exceeded = True
                process.kill()
                break

            if time.time() - start_time > timeout_seconds:
                timed_out = True
                process.kill()
                break

            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)

        process.join(1)

        if len(output) != 0:
            return output[0]

        elapsed_seconds = time.time() - start_time
        if timed_out:
            return _process_failed_result(
                in_outs,
                error_message="Global Timeout",
                error_code=-3,
                exitcode=process.exitcode,
                elapsed_seconds=elapsed_seconds,
                observed_memory_bytes=observed_memory_bytes,
                memory_limit_bytes=memory_limit_bytes,
            )
        if memory_limit_exceeded:
            return _process_failed_result(
                in_outs,
                error_message="Memory Limit Exceeded",
                error_code=-4,
                exitcode=process.exitcode,
                elapsed_seconds=elapsed_seconds,
                observed_memory_bytes=observed_memory_bytes,
                memory_limit_bytes=memory_limit_bytes,
            )
        return _process_failed_result(
            in_outs,
            error_message="Process failed",
            error_code=-5,
            exitcode=process.exitcode,
            elapsed_seconds=elapsed_seconds,
            observed_memory_bytes=observed_memory_bytes,
            memory_limit_bytes=memory_limit_bytes,
        )
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        manager.shutdown()


def _is_test_success(value: Any) -> bool:
    if isinstance(value, np.bool_):
        return bool(value)
    return value is True


def _normalize_test_results(results: list[Any]) -> list[Any]:
    normalized = []
    for item in results:
        if isinstance(item, np.ndarray):
            item = item.item(0)
        if isinstance(item, np.bool_):
            item = bool(item)
        normalized.append(item)
    return normalized


def summarize_test_results(results: list[Any]) -> dict[str, Any]:
    normalized_results = _normalize_test_results(results)
    num_tests = len(normalized_results)
    num_passed = sum(1 for item in normalized_results if _is_test_success(item))
    pass_rate = 1.0 if num_tests == 0 else float(num_passed / num_tests)
    return {
        "results": normalized_results,
        "num_tests": num_tests,
        "num_passed": num_passed,
        "pass_rate": pass_rate,
    }


def evaluate_code_against_in_outs(
    solution_str: str,
    in_outs: dict[str, Any],
    *,
    timeout: int = 6,
    complete_evaluation: bool = False,
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(solution_str, str):
        raise TypeError(f"solution_str must be a string, got {type(solution_str)}")

    code = extract_code_from_response(solution_str)
    results, detail = check_correctness(
        in_outs=in_outs,
        code=code,
        timeout=timeout,
        complete_evaluation=complete_evaluation,
    )
    results = _normalize_test_results(results)
    success = all(_is_test_success(item) for item in results)
    return success, {"results": results, "detail": detail}


def evaluate_response(question_id: str, solution_str: str, benchmark_dir: str, timeout: int = 6) -> tuple[bool, dict[str, Any]]:
    benchmark = load_benchmark(question_id, benchmark_dir)
    in_outs = build_in_outs(benchmark)
    success, metadata = evaluate_code_against_in_outs(solution_str, in_outs, timeout=timeout)
    metadata = {
        "question_id": question_id,
        "results": metadata["results"],
        "detail": metadata["detail"],
    }
    return success, metadata
