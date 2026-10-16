"""Prepare the TACO test split into the recipe-local LCB-v6 benchmark format.

This script replays the TCS-filtered TACO test pkl published at
  - https://huggingface.co/datasets/NeurIPS20403/TACO_Eval
and emits:
  - `<output_dir>/test_prepared.parquet` — generation parquet with the same
    schema as `eval/livecodebench_v6/shared/test6.parquet`, with sample outputs
    2+ already masked in the user prompt content.
  - `<output_dir>/benchmark_json/<question_id>.json` — per-question benchmark
    record compatible with `recipe.code_bench.livecodebench.
    codegen_problem_from_json_dict` / `load_benchmark`. Each JSON carries the
    `public_test_cases` (parsed from markdown) and `private_test_cases` (hidden
    inputs/outputs with the public-test overlap removed).
  - `<output_dir>/test_sanity_report.json` — summary of extraction success,
    format distribution, public/private overlap statistics, and the list of
    discarded question indices (no recognized sample block).

Usage:
    python -m recipe.code_bench.prepare_taco \\
        --output-dir eval/taco/shared \\
        [--test-pkl <path>] [--force]

When `--test-pkl` is omitted the script downloads it from the HF hub into
`<output_dir>/raw/` via `huggingface_hub.hf_hub_download`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Shim: the TCS-published pkl carries pandas Timestamp NaT objects pickled with
# an older pandas that exposed `__nat_unpickle`. Newer pandas removed the
# attribute but keeps NaTType identical, so we inject a no-op constructor.
import pandas._libs.tslibs.nattype as _nt  # noqa: E402

if not hasattr(_nt, "__nat_unpickle"):
    _nt.__nat_unpickle = lambda *_a, **_k: _nt.NaTType()  # type: ignore[attr-defined]

import pandas as pd  # noqa: E402

from recipe.code_bench.livecodebench import (  # noqa: E402
    CodeGenerationProblem,
    Difficulty,
    Platform,
    Test,
    TestType,
    codegen_problem_to_json_dict,
    format_prompt_generation,
)
from recipe.code_bench.taco_sample_extraction import (  # noqa: E402
    detect_format,
    extract_samples,
    mask_markdown_sample_outputs,
    remove_public_overlap,
)


_TEST_REPO = "NeurIPS20403/TACO_Eval"
_TEST_FILENAME = "taco_for_test_evaluation.pkl"


# ---------------------------------------------------------------------------
# Source / difficulty mapping onto CodeGenerationProblem enums
# ---------------------------------------------------------------------------


_SOURCE_TO_PLATFORM = {
    "codeforces": Platform.CODEFORCES,
    "atcoder": Platform.ATCODER,
    "aizu": Platform.AIZU,
    "codechef": Platform.CODECHEF,
    "hackerearth": Platform.HACKEREARTH,
    "hackerrank": Platform.HACKEREARTH,  # HackerEarth enum reused; TACO test uses both labels
    "codewars": Platform.CODEWARS,
    "geeksforgeeks": Platform.GEEKSFORGEEKS,
    "leetcode": Platform.LEETCODE,
}


_DIFFICULTY_MAP = {
    "EASY": Difficulty.EASY,
    "MEDIUM": Difficulty.MEDIUM,
    "MEDIUM_HARD": Difficulty.MEDIUM_HARD,
    "HARD": Difficulty.HARD,
    "VERY_HARD": Difficulty.VERY_HARD,
    "UNKNOWN_DIFFICULTY": Difficulty.UNKNOWN_DIFFICULTY,
}


def _resolve_platform(source: str | None) -> Platform:
    if source is None:
        return Platform.TACO_OTHER
    return _SOURCE_TO_PLATFORM.get(source.lower(), Platform.TACO_OTHER)


def _resolve_difficulty(difficulty: str | None) -> Difficulty:
    if difficulty is None:
        return Difficulty.UNKNOWN_DIFFICULTY
    return _DIFFICULTY_MAP.get(str(difficulty).upper(), Difficulty.UNKNOWN_DIFFICULTY)


# ---------------------------------------------------------------------------
# Per-problem processing
# ---------------------------------------------------------------------------


def _build_question_id(source: str | None, split: str, stable_index: int) -> str:
    src = (source or "unknown").lower()
    return f"taco-{split}-{src}-{int(stable_index):06d}"


def _load_pkl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, list):
        raise TypeError(f"Expected TACO pkl to be a list, got {type(payload)}")
    return payload


def _download_if_missing(local_path: Path, repo_id: str, filename: str) -> Path:
    if local_path.is_file():
        return local_path
    from huggingface_hub import hf_hub_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_path.parent),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != local_path:
        # hf_hub_download may place the file under local_dir/<filename>.
        target = local_path.parent / filename
        if not target.is_file():
            raise FileNotFoundError(f"hf_hub_download did not produce expected file at {target}")
        return target
    return downloaded_path


def _build_problem(
    *,
    question_id: str,
    raw_item: dict[str, Any],
    public_tests: list[dict[str, str]],
    private_inputs: list[str],
    private_outputs: list[str],
    masked_question_content: str,
) -> CodeGenerationProblem:
    extra = raw_item.get("extra_info", {}) or {}
    source = extra.get("source")
    difficulty = extra.get("difficulty")

    public_test_cases = tuple(
        Test(input=sample["input"], output=sample["output"], testtype=TestType.STDIN)
        for sample in public_tests
    )
    private_test_cases = tuple(
        Test(input=inp, output=out, testtype=TestType.STDIN)
        for inp, out in zip(private_inputs, private_outputs, strict=True)
    )

    # TACO-filtered subset is entirely stdin (no fn_name); keep starter_code empty.
    contest_date = extra.get("date")
    contest_date_iso = (
        str(contest_date)
        if contest_date is not None and str(contest_date).strip() and str(contest_date) != "NaT"
        else "1970-01-01T00:00:00"
    )
    try:
        from datetime import datetime

        contest_date_parsed = datetime.fromisoformat(contest_date_iso)
    except ValueError:
        contest_date_parsed = datetime.fromisoformat("1970-01-01T00:00:00")

    return CodeGenerationProblem(
        question_title=str(extra.get("question_title") or extra.get("name") or question_id),
        question_content=masked_question_content,
        platform=_resolve_platform(source),
        question_id=question_id,
        contest_id=str(extra.get("name") or ""),
        contest_date=contest_date_parsed,
        starter_code="",
        difficulty=_resolve_difficulty(difficulty),
        public_test_cases=public_test_cases,
        private_test_cases=private_test_cases,
        metadata={"source": source, "url": extra.get("url"), "tags": extra.get("tags")},
    )


def _build_generation_record(problem: CodeGenerationProblem, split: str) -> dict[str, Any]:
    """Generation-parquet row. Mirrors `prepare_livecodebench_v6.build_generation_record`.

    `data_source` is the single literal `"taco"` so metrics aggregate across all
    difficulty tiers; the per-problem tier is kept in the `difficulty` column.
    """
    from datetime import datetime

    safe_date = (
        problem.contest_date.isoformat()
        if isinstance(problem.contest_date, datetime)
        else str(problem.contest_date)
    )
    difficulty_value = (
        problem.difficulty.value
        if hasattr(problem.difficulty, "value")
        else str(problem.difficulty)
    )
    return {
        "data_source": "taco",
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "platform": problem.platform.value,
        "difficulty": difficulty_value,
        "contest_date": safe_date,
        "split": split,
        "prompt": format_prompt_generation(problem),
        "reward_model": {
            "style": "rule",
            "ground_truth": {"question_id": problem.question_id},
        },
    }


# ---------------------------------------------------------------------------
# Top-level split preparation
# ---------------------------------------------------------------------------


def _prepare_split(
    *,
    pkl_path: Path,
    output_dir: Path,
    split: str,
    log_every: int = 1000,
) -> dict[str, Any]:
    print(f"[{split}] loading {pkl_path}...", file=sys.stderr)
    raw_items = _load_pkl(pkl_path)
    print(f"[{split}] loaded {len(raw_items)} items", file=sys.stderr)

    # Question IDs are prefixed with the split (taco-<split>-<source>-<idx>).
    benchmark_dir = output_dir / "benchmark_json"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    format_counter: Counter[str | None] = Counter()
    source_counter: Counter[str] = Counter()
    discard_counter: Counter[str] = Counter()
    n_public_dist: Counter[int] = Counter()
    n_private_dist: list[int] = []
    overlap_counter: list[int] = []
    discarded_ids: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()

    for stable_idx, item in enumerate(raw_items):
        extra = item.get("extra_info", {}) or {}
        source = extra.get("source") or "unknown"
        source_counter[source] += 1
        markdown = extra.get("question") or ""

        fmt_tag = detect_format(markdown)
        format_counter[fmt_tag] += 1

        public_samples = extract_samples(markdown)
        if not public_samples:
            discard_counter["no_public_samples"] += 1
            discarded_ids.append({"index": stable_idx, "source": source, "reason": "no_public_samples"})
            continue

        gt = item["reward_model"]["ground_truth"]
        gt_inputs = list(gt.get("inputs") or [])
        gt_outputs = list(gt.get("outputs") or [])
        if len(gt_inputs) != len(gt_outputs):
            discard_counter["inputs_outputs_length_mismatch"] += 1
            discarded_ids.append(
                {"index": stable_idx, "source": source, "reason": "inputs_outputs_length_mismatch"}
            )
            continue

        private_inputs, private_outputs, n_overlap = remove_public_overlap(
            public_samples, gt_inputs, gt_outputs
        )
        if len(private_inputs) == 0:
            discard_counter["no_private_after_dedup"] += 1
            discarded_ids.append(
                {"index": stable_idx, "source": source, "reason": "no_private_after_dedup"}
            )
            continue

        masked_markdown = mask_markdown_sample_outputs(markdown)

        question_id = _build_question_id(source, split, stable_idx)
        if question_id in seen_question_ids:
            discard_counter["duplicate_question_id"] += 1
            discarded_ids.append(
                {"index": stable_idx, "source": source, "reason": "duplicate_question_id"}
            )
            continue
        seen_question_ids.add(question_id)

        problem = _build_problem(
            question_id=question_id,
            raw_item=item,
            public_tests=public_samples,
            private_inputs=private_inputs,
            private_outputs=private_outputs,
            masked_question_content=masked_markdown,
        )

        benchmark_path = benchmark_dir / f"{question_id}.json"
        with benchmark_path.open("w", encoding="utf-8") as f:
            json.dump(codegen_problem_to_json_dict(problem), f, ensure_ascii=False)

        records.append(_build_generation_record(problem, split))
        n_public_dist[len(public_samples)] += 1
        n_private_dist.append(len(private_inputs))
        overlap_counter.append(n_overlap)

        if (stable_idx + 1) % log_every == 0:
            print(
                f"[{split}] processed {stable_idx + 1}/{len(raw_items)} "
                f"(kept {len(records)}, discarded {sum(discard_counter.values())})",
                file=sys.stderr,
            )

    out_parquet = output_dir / f"{split}_prepared.parquet"
    pd.DataFrame.from_records(records).to_parquet(out_parquet, index=False)

    sanity = {
        "split": split,
        "total_input_items": len(raw_items),
        "total_kept": len(records),
        "format_distribution": {str(k): v for k, v in format_counter.items()},
        "source_distribution": dict(source_counter),
        "discard_distribution": dict(discard_counter),
        "n_public_tests_distribution": dict(n_public_dist),
        "n_private_tests_mean": (
            sum(n_private_dist) / max(len(n_private_dist), 1) if n_private_dist else 0.0
        ),
        "n_private_tests_median": (
            sorted(n_private_dist)[len(n_private_dist) // 2] if n_private_dist else 0
        ),
        "n_private_tests_max": max(n_private_dist) if n_private_dist else 0,
        "overlap_n_dropped_mean": (
            sum(overlap_counter) / max(len(overlap_counter), 1) if overlap_counter else 0.0
        ),
        "output_parquet": str(out_parquet),
        "benchmark_dir": str(benchmark_dir),
        "first_20_discarded": discarded_ids[:20],
    }
    report_path = output_dir / f"{split}_sanity_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(sanity, f, ensure_ascii=False, indent=2)

    extract_rate = len(records) / max(len(raw_items), 1)
    print(
        f"[{split}] kept {len(records)}/{len(raw_items)} ({extract_rate * 100:.1f}%); "
        f"parquet -> {out_parquet}; benchmarks -> {benchmark_dir}",
        file=sys.stderr,
    )
    return sanity


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("eval/taco/shared"))
    parser.add_argument("--test-pkl", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even if prepared parquet already exists.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    test_pkl = args.test_pkl or _download_if_missing(
        output_dir / "raw" / _TEST_FILENAME, _TEST_REPO, _TEST_FILENAME
    )
    test_parquet = output_dir / "test_prepared.parquet"
    if test_parquet.exists() and not args.force:
        raise FileExistsError(f"{test_parquet} already exists; pass --force to re-process.")

    sanity = _prepare_split(pkl_path=test_pkl, output_dir=output_dir, split="test")
    summary_path = output_dir / "prepare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"splits": ["test"], "per_split": {"test": sanity}}, f, ensure_ascii=False, indent=2)
    print(f"summary -> {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
