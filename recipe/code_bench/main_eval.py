"""
Compute strict pass@k plus public-test BON metrics for sampled code responses.

Use this script after code generation to score each sampled response against the
benchmark JSON files. It writes a `_pass.csv` summary plus a scored parquet with
strict evaluation detail and public/private split pass flags.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from recipe.code_bench.livecodebench import derive_public_private_pass_flags, load_codegen_problem
from recipe.code_bench.reward_score import evaluate_response


EVAL_TASKS_PER_WORKER_PER_CHUNK = 64
EVAL_WORKERS_HARD_CAP = 32


def _default_max_workers() -> int:
    """cgroup-aware: min(cgroup_cpus // 2, EVAL_WORKERS_HARD_CAP). os.cpu_count() returns host cores and oversubscribes on k8s, triggering the pipe-fill deadlock the chunking is meant to dodge."""
    cgroup_v2 = Path("/sys/fs/cgroup/cpu.max")
    if cgroup_v2.exists():
        quota_s, period_s = cgroup_v2.read_text().split()
        cpus = (
            os.cpu_count() or 1 if quota_s == "max"
            else max(1, int(quota_s) // int(period_s))
        )
    else:
        v1_q = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        v1_p = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if v1_q.exists() and v1_p.exists():
            quota = int(v1_q.read_text())
            period = int(v1_p.read_text())
            cpus = os.cpu_count() or 1 if quota <= 0 else max(1, quota // period)
        else:
            cpus = os.cpu_count() or 1
    return max(1, min(cpus // 2, EVAL_WORKERS_HARD_CAP))


def comb_estimator(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def bon_public_estimator(public_passes: list[int], private_passes: list[int], n: int) -> float:
    if n < 1:
        raise ValueError(f"bon_n values must be positive integers, got {n}")
    if n > len(public_passes):
        raise ValueError(f"bon_n={n} cannot exceed the number of samples per problem={len(public_passes)}")

    public_prefix = public_passes[:n]
    private_prefix = private_passes[:n]
    public_candidates = int(sum(public_prefix))
    if public_candidates == 0:
        return 0.0

    selected_and_private_passed = sum(
        1 for public_pass, private_pass in zip(public_prefix, private_prefix, strict=True) if public_pass and private_pass
    )
    return float(selected_and_private_passed / public_candidates)


def parse_pass_k_list(raw_value: str, n_samples: int) -> list[int]:
    if not raw_value:
        return list(range(1, n_samples + 1))

    pass_ks = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not pass_ks:
        raise ValueError("pass_k_list must contain at least one integer")
    if len(set(pass_ks)) != len(pass_ks):
        raise ValueError(f"pass_k_list contains duplicates: {pass_ks}")
    if any(k < 1 for k in pass_ks):
        raise ValueError(f"pass_k_list must contain only positive integers: {pass_ks}")
    if any(k > n_samples for k in pass_ks):
        raise ValueError(f"pass_k_list cannot contain values larger than n_samples={n_samples}: {pass_ks}")
    return pass_ks


@lru_cache(maxsize=None)
def load_benchmark_problem(question_id: str, benchmark_dir: str):
    return load_codegen_problem(Path(benchmark_dir) / f"{question_id}.json")


def normalize_response_list(responses: object) -> list[str]:
    if isinstance(responses, np.ndarray):
        normalized = responses.tolist()
    elif hasattr(responses, "tolist") and not isinstance(responses, (list, str)):
        normalized = responses.tolist()
    else:
        normalized = responses

    if not isinstance(normalized, list):
        raise TypeError(f"responses must be a list-like object, got {type(responses)}")
    if any(not isinstance(item, str) for item in normalized):
        raise TypeError("responses must contain only strings")
    return normalized


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {type(key)}")
            result[key] = to_jsonable(item)
        return result
    raise TypeError(f"Unsupported JSON value type: {type(value)}")


def default_scored_output_path(data_path: Path, pass_at_1: float) -> Path:
    score_suffix = f"{100.0 * pass_at_1:.2f}"
    if data_path.suffix:
        return data_path.with_name(f"{data_path.stem}_{score_suffix}{data_path.suffix}")
    return data_path.with_name(f"{data_path.name}_{score_suffix}.parquet")


def _drive_pool_with_backpressure(
    executor: ProcessPoolExecutor,
    fn: Callable[..., Any],
    tasks: Iterable[tuple[Any, ...]],
    on_result: Callable[..., None],
    progress: "Any",
    pending_limit: int,
) -> None:
    """Submit tasks to `executor` with bounded in-flight count.

    Burst-submitting many thousand tasks at once into ProcessPoolExecutor on
    Linux (Python 3.12) deadlocks for large benches: parent's wakeup pipe and
    workers' result pipe both fill up before the manager thread can drain
    them. Holding in-flight at `pending_limit` keeps both pipes well below
    capacity. Bench-by-bench impact: TACO test (881 problems × 16 = 14096
    tasks) reproducibly hangs without this; LCB v6 (175 × 16 = 2800) and
    CodeContests (239 × 16 = 3824) happen to fit under the threshold.
    """
    in_flight: dict[Any, tuple[Any, ...]] = {}
    task_iter = iter(tasks)
    for _ in range(pending_limit):
        try:
            task = next(task_iter)
        except StopIteration:
            break
        in_flight[executor.submit(fn, *task)] = task
    while in_flight:
        done_set, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        for done_future in done_set:
            on_result(*done_future.result())
            del in_flight[done_future]
            progress.update(1)
            try:
                task = next(task_iter)
            except StopIteration:
                continue
            in_flight[executor.submit(fn, *task)] = task


def _run_in_chunks(
    fn: Callable[..., Any],
    tasks: list[tuple[Any, ...]],
    on_result: Callable[..., None],
    *,
    max_workers: int,
    chunk_size: int,
    pending_limit: int,
    desc: str | None = None,
) -> None:
    """Per-chunk fresh ProcessPoolExecutor; chunk_size sized so each worker handles ~chunk_size/max_workers tasks before the pool exits and is rebuilt. testing_util.py exec()s candidate code in-process and worker globals never get cleared, so chunk-level pool rebuild is what bounds the per-worker memory growth (chose this over ProcessPoolExecutor's own max_tasks_per_child to avoid a Python 3.12 race that surfaces as EOFError when a worker self-terminates while futures are in flight)."""
    total = len(tasks)
    with tqdm(total=total, desc=desc) as progress:
        for start in range(0, total, chunk_size):
            chunk = tasks[start:start + chunk_size]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                _drive_pool_with_backpressure(
                    executor, fn, chunk, on_result, progress,
                    pending_limit=pending_limit,
                )


def score_single_response(
    row_idx: int,
    sample_idx: int,
    question_id: str,
    response: str,
    benchmark_dir: str,
    timeout: int,
) -> tuple[int, int, int, dict[str, Any]]:
    success, metadata = evaluate_response(
        question_id=question_id,
        solution_str=response,
        benchmark_dir=benchmark_dir,
        timeout=timeout,
    )
    return row_idx, sample_idx, int(success), to_jsonable(metadata)


def default_output_csv(data_path: Path) -> Path:
    if data_path.suffix:
        return data_path.with_name(f"{data_path.stem}_pass.csv")
    return data_path.with_name(f"{data_path.name}_pass.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=str, required=True)
    parser.add_argument("--response-key", type=str, default="responses")
    parser.add_argument("--question-id-key", type=str, default="question_id")
    parser.add_argument("--pass-k-list", type=str, default="")
    parser.add_argument("--bon-n-list", type=str, default="")
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--max-workers", type=int, default=_default_max_workers())
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help=f"Per-chunk task count; each chunk gets a fresh ProcessPoolExecutor. Default = max(max_workers * {EVAL_TASKS_PER_WORKER_PER_CHUNK}, 256), so each worker handles ~{EVAL_TASKS_PER_WORKER_PER_CHUNK} tasks per pool lifetime to bound exec()-leaked globals.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {args.max_workers}")

    pf = pq.ParquetFile(args.data_path)
    dataset = pd.concat(
        [b.to_pandas() for b in pf.iter_batches(batch_size=64)],
        ignore_index=True,
    )
    if len(dataset) == 0:
        raise ValueError(f"Input parquet is empty: {args.data_path}")

    if args.response_key not in dataset.columns:
        raise KeyError(f"Missing response column: {args.response_key}")
    if args.question_id_key not in dataset.columns:
        raise KeyError(f"Missing question_id column: {args.question_id_key}")

    response_lists = [normalize_response_list(responses) for responses in dataset[args.response_key].tolist()]
    question_ids = dataset[args.question_id_key].tolist()

    if any(not isinstance(question_id, str) or not question_id for question_id in question_ids):
        raise ValueError(f"{args.question_id_key} must contain non-empty strings")

    n_samples = len(response_lists[0])
    if n_samples == 0:
        raise ValueError("responses must contain at least one sample per prompt")
    if any(len(responses) != n_samples for responses in response_lists):
        raise ValueError("All rows must contain the same number of sampled responses")

    pass_ks = parse_pass_k_list(args.pass_k_list, n_samples)
    bon_ns = parse_pass_k_list(args.bon_n_list, n_samples) if args.bon_n_list else pass_ks
    score_matrix = np.zeros((len(dataset), n_samples), dtype=np.int8)
    detail_matrix: list[list[dict[str, Any] | None]] = [[None for _ in range(n_samples)] for _ in range(len(dataset))]

    eval_tasks = [
        (
            row_idx,
            sample_idx,
            question_id,
            response,
            args.benchmark_dir,
            args.timeout,
        )
        for row_idx, (question_id, responses) in enumerate(zip(question_ids, response_lists, strict=True))
        for sample_idx, response in enumerate(responses)
    ]
    pending_limit = max(args.max_workers * 2, 64)
    if args.chunk_size is None:
        args.chunk_size = max(args.max_workers * EVAL_TASKS_PER_WORKER_PER_CHUNK, 256)
    if args.chunk_size < pending_limit:
        raise ValueError(
            f"--chunk-size={args.chunk_size} must be >= pending_limit={pending_limit} "
            f"(= max(max_workers*2, 64)) so backpressure can fully prefill each chunk's pool"
        )

    def _record_score(row_idx: int, sample_idx: int, success: int, detail: dict[str, Any]) -> None:
        score_matrix[row_idx, sample_idx] = success
        detail_matrix[row_idx][sample_idx] = detail

    _run_in_chunks(
        score_single_response, eval_tasks, _record_score,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        pending_limit=pending_limit,
        desc="strict_eval",
    )

    num_correct = score_matrix.sum(axis=1).astype(int)
    output_csv = args.output_csv or default_output_csv(args.data_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "data_path": str(args.data_path),
        "benchmark_dir": args.benchmark_dir,
        "num_problems": int(len(dataset)),
        "n_samples": int(n_samples),
        "pass@1": float(score_matrix.mean()),
        "first_sample_pass@1": float(score_matrix[:, 0].mean()),
    }
    row.update(
        {
            f"pass@{k}": float(np.mean([comb_estimator(n_samples, c, k) for c in num_correct]))
            for k in pass_ks
            if k != 1
        }
    )
    sample_passes = score_matrix.tolist()
    sample_public_passes: list[list[int]] = []
    sample_private_passes: list[list[int]] = []
    sample_details_json: list[list[str]] = []
    bon_metric_rows: dict[str, list[float]] = {f"BON@{n}_public": [] for n in bon_ns}
    for row_idx, (question_id, row_details) in enumerate(zip(question_ids, detail_matrix, strict=True)):
        if any(detail is None for detail in row_details):
            raise RuntimeError(f"Missing evaluation detail for row_idx={row_idx}")
        problem = load_benchmark_problem(question_id, args.benchmark_dir)

        row_public_passes: list[int] = []
        row_private_passes: list[int] = []
        row_details_json: list[str] = []
        for sample_idx, detail in enumerate(row_details):
            if detail is None:
                raise RuntimeError(f"Missing evaluation detail for question_id={question_id}, sample_idx={sample_idx}")
            public_pass, private_pass = derive_public_private_pass_flags(problem, detail["results"])
            row_public_passes.append(int(public_pass))
            row_private_passes.append(int(private_pass))
            row_details_json.append(json.dumps(detail, ensure_ascii=False))

        sample_public_passes.append(row_public_passes)
        sample_private_passes.append(row_private_passes)
        sample_details_json.append(row_details_json)

        for bon_n in bon_ns:
            bon_metric_rows[f"BON@{bon_n}_public"].append(
                bon_public_estimator(row_public_passes, row_private_passes, bon_n)
            )

    pass_at_1 = float(row.get("pass@1", row["first_sample_pass@1"]))
    scored_output_path = default_scored_output_path(args.data_path, pass_at_1)
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_dataset = dataset.copy()
    scored_dataset["sample_passes"] = sample_passes
    scored_dataset["sample_public_passes"] = sample_public_passes
    scored_dataset["sample_private_passes"] = sample_private_passes
    scored_dataset["sample_details_json"] = sample_details_json
    for metric_name, values in bon_metric_rows.items():
        scored_dataset[metric_name] = values
        row[metric_name] = float(np.mean(values))
    pd.DataFrame([row]).to_csv(output_csv, index=False)
    scored_dataset.to_parquet(scored_output_path, index=False)

    for key, value in row.items():
        print(f"{key}: {value}")
    print(f"Wrote pass@k summary to {output_csv}")
    print(f"Wrote scored parquet to {scored_output_path}")


if __name__ == "__main__":
    main()
