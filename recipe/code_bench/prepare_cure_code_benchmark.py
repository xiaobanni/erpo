"""Prepare CURE-released code benchmarks into the recipe-local code format.

This script converts the CURE project evaluation datasets:
  - Gen-Verse/CodeContests
  - Gen-Verse/CodeForces

into the same recipe-local benchmark shape already used by LiveCodeBench v6 and
TACO:
  - <output_dir>/test_prepared.parquet
  - <output_dir>/benchmark_json/<question_id>.json
  - <output_dir>/test_sanity_report.json

The resulting parquet / benchmark JSONs can be consumed directly by:
  - recipe.code_bench.main_eval (transfer evaluation)
  - recipe.code_bench.reward_score
  - recipe.code_bench.test_rl_reward_manager

Usage:
    python -m recipe.code_bench.prepare_cure_code_benchmark \\
        --dataset codeforces \\
        --output-dir eval/codeforces/shared
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from recipe.code_bench.create_hidden_output_parquet import hide_sample_outputs
from recipe.code_bench.livecodebench import (
    CodeGenerationProblem,
    Difficulty,
    Platform,
    Test,
    TestType,
    codegen_problem_to_json_dict,
    format_prompt_generation,
)


_DATASET_CONFIG = {
    "codeforces": {
        "hf_repo": "Gen-Verse/CodeForces",
        "hf_filename": "test/CodeForces.json",
        "default_output_dir": Path("eval/codeforces/shared"),
        "default_platform": Platform.CODEFORCES,
        "data_source": "codeforces",
    },
    "codecontests": {
        "hf_repo": "Gen-Verse/CodeContests",
        "hf_filename": "test/CodeContests.json",
        "default_output_dir": Path("eval/codecontests/shared"),
        "default_platform": Platform.TACO_OTHER,
        "data_source": "codecontests",
    },
}

_PLATFORM_MAP = {
    "codeforces": Platform.CODEFORCES,
    "atcoder": Platform.ATCODER,
    "codechef": Platform.CODECHEF,
    "hackerearth": Platform.HACKEREARTH,
    "hackerrank": Platform.HACKEREARTH,
    "aizu": Platform.AIZU,
    "codewars": Platform.CODEWARS,
    "geeksforgeeks": Platform.GEEKSFORGEEKS,
    "leetcode": Platform.LEETCODE,
}

_DIFFICULTY_MAP = {
    "easy": Difficulty.EASY,
    "medium": Difficulty.MEDIUM,
    "hard": Difficulty.HARD,
    "medium_hard": Difficulty.MEDIUM_HARD,
    "very_hard": Difficulty.VERY_HARD,
    "unknown_difficulty": Difficulty.UNKNOWN_DIFFICULTY,
}

_TITLE_KEYS = (
    "question_title",
    "title",
    "name",
    "problem_name",
    "problem_title",
    "contest_problem_name",
)

_ID_KEYS = (
    "question_id",
    "problem_id",
    "task_id",
    "id",
    "name",
    "slug",
    "cf_contest_problem_id",
)

_DATE_KEYS = (
    "contest_date",
    "date",
    "created_at",
    "creation_time",
    "time",
)


def _download_if_missing(local_path: Path, repo_id: str, filename: str) -> Path:
    if local_path.is_file():
        return local_path

    from huggingface_hub import hf_hub_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=str(local_path.parent),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != local_path:
        candidate = local_path.parent / Path(filename).name
        if candidate.is_file():
            return candidate
    return downloaded_path


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise TypeError(f"Expected dataset JSON to be a list, got {type(payload)}")
    if any(not isinstance(item, dict) for item in payload):
        raise TypeError("Dataset JSON must be a list of objects")
    return payload


def _coerce_string_list(raw_value: Any, field_name: str) -> list[str]:
    if raw_value is None:
        raise ValueError(f"{field_name} is missing")
    if not isinstance(raw_value, list):
        raise TypeError(f"{field_name} must be a list, got {type(raw_value)}")
    values: list[str] = []
    for idx, item in enumerate(raw_value):
        if isinstance(item, str):
            values.append(item)
        elif item is None:
            raise TypeError(f"{field_name}[{idx}] is null")
        else:
            values.append(str(item))
    return values


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _find_first_string(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return None


def _parse_date(item: dict[str, Any]) -> datetime:
    for key in _DATE_KEYS:
        raw_value = item.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, (int, float)):
            try:
                return datetime.fromtimestamp(float(raw_value))
            except (OverflowError, OSError, ValueError):
                continue
        if isinstance(raw_value, str) and raw_value.strip():
            candidate = raw_value.strip()
            try:
                return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            except ValueError:
                pass
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
    return datetime.fromisoformat("1970-01-01T00:00:00")


def _infer_platform(dataset_name: str, item: dict[str, Any]) -> Platform:
    source = None
    for key in ("platform", "source", "oj_name", "origin_platform"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            source = value.strip().lower()
            break
    if source is None:
        return _DATASET_CONFIG[dataset_name]["default_platform"]
    return _PLATFORM_MAP.get(source, _DATASET_CONFIG[dataset_name]["default_platform"])


def _infer_difficulty(item: dict[str, Any]) -> Difficulty:
    raw_difficulty = item.get("difficulty")
    if isinstance(raw_difficulty, str) and raw_difficulty.strip():
        key = raw_difficulty.strip().lower()
        if key in _DIFFICULTY_MAP:
            return _DIFFICULTY_MAP[key]

    rating = item.get("rating")
    if isinstance(rating, (int, float)):
        rating_value = float(rating)
        if rating_value <= 1200:
            return Difficulty.EASY
        if rating_value <= 1800:
            return Difficulty.MEDIUM
        return Difficulty.HARD

    return Difficulty.UNKNOWN_DIFFICULTY


def _infer_title(item: dict[str, Any], question: str, index: int) -> str:
    explicit = _find_first_string(item, _TITLE_KEYS)
    if explicit is not None:
        return explicit
    first_line = next((line.strip() for line in question.splitlines() if line.strip()), "")
    if first_line:
        return first_line[:160]
    return f"Problem {index}"


def _question_contains_public_examples(question: str, public_tests: list[Test]) -> bool:
    if not public_tests:
        return True
    normalized_question = _normalize_text(question)
    for test in public_tests:
        normalized_input = _normalize_text(test.input)
        normalized_output = _normalize_text(test.output)
        if normalized_input and normalized_input not in normalized_question:
            return False
        if normalized_output and normalized_output not in normalized_question:
            return False
    return True


def _render_public_examples(public_tests: list[Test]) -> str:
    if not public_tests:
        return ""
    sections = ["### Public Examples"]
    for idx, test in enumerate(public_tests, start=1):
        sections.append(f"Example {idx} Input:")
        sections.append(test.input)
        sections.append(f"Example {idx} Output:")
        sections.append(test.output)
    return "\n".join(sections)


def _build_question_content(
    question: str,
    public_tests: list[Test],
    embed_public_examples: str,
) -> tuple[str, bool]:
    if embed_public_examples == "never":
        return question, False
    if embed_public_examples == "always":
        rendered = _render_public_examples(public_tests)
        if not rendered:
            return question, False
        return f"{question.rstrip()}\n\n{rendered}\n", True

    if _question_contains_public_examples(question, public_tests):
        return question, False

    rendered = _render_public_examples(public_tests)
    if not rendered:
        return question, False
    return f"{question.rstrip()}\n\n{rendered}\n", True


def _build_question_id(dataset_name: str, index: int) -> str:
    return f"{dataset_name}-test-{index:06d}"


def _build_problem(
    *,
    dataset_name: str,
    item: dict[str, Any],
    index: int,
    question_content: str,
    public_tests: list[Test],
    private_tests: list[Test],
) -> CodeGenerationProblem:
    question = item["question"]
    title = _infer_title(item, question, index)
    explicit_id = _find_first_string(item, _ID_KEYS)
    platform = _infer_platform(dataset_name, item)
    difficulty = _infer_difficulty(item)
    contest_date = _parse_date(item)
    question_id = _build_question_id(dataset_name, index)

    metadata = {
        "source_dataset": dataset_name,
        "original_id": explicit_id,
        "platform_raw": item.get("platform") or item.get("source"),
        "rating": item.get("rating"),
        "tags": item.get("tags"),
        "public_example_count": len(public_tests),
        "private_test_count": len(private_tests),
    }

    return CodeGenerationProblem(
        question_title=title,
        question_content=question_content,
        platform=platform,
        question_id=question_id,
        contest_id=explicit_id or dataset_name,
        contest_date=contest_date,
        starter_code="",
        difficulty=difficulty,
        public_test_cases=tuple(public_tests),
        private_test_cases=tuple(private_tests),
        metadata=metadata,
    )


def _build_generation_record(problem: CodeGenerationProblem, data_source: str) -> dict[str, Any]:
    return {
        "data_source": data_source,
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "platform": problem.platform.value,
        "difficulty": problem.difficulty.value,
        "contest_date": problem.contest_date.isoformat(),
        "prompt": format_prompt_generation(problem),
        "reward_model": {
            "style": "rule",
            "ground_truth": {"question_id": problem.question_id},
        },
    }


def prepare_cure_code_benchmark(
    *,
    dataset_name: str,
    input_json: Path | None,
    output_dir: Path,
    embed_public_examples: str,
    mask_additional_sample_outputs: bool,
    force: bool,
) -> dict[str, Any]:
    config = _DATASET_CONFIG[dataset_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    local_json = input_json or (raw_dir / Path(config["hf_filename"]).name)
    resolved_input = _download_if_missing(local_json, config["hf_repo"], config["hf_filename"])

    out_parquet = output_dir / "test_prepared.parquet"
    benchmark_dir = output_dir / "benchmark_json"
    sanity_path = output_dir / "test_sanity_report.json"

    if force:
        if benchmark_dir.is_dir():
            shutil.rmtree(benchmark_dir)
    elif out_parquet.exists() or benchmark_dir.exists() or sanity_path.exists():
        raise FileExistsError(
            f"Output already exists under {output_dir}. Pass --force to overwrite."
        )

    benchmark_dir.mkdir(parents=True, exist_ok=True)

    raw_items = _load_json(resolved_input)
    records: list[dict[str, Any]] = []
    platform_counter: Counter[str] = Counter()
    difficulty_counter: Counter[str] = Counter()
    discard_counter: Counter[str] = Counter()
    embedded_counter: Counter[str] = Counter()
    mask_counter: Counter[str] = Counter()
    public_source_counter: Counter[str] = Counter()
    public_counts: list[int] = []
    private_counts: list[int] = []
    discarded_indices: list[int] = []

    for index, item in enumerate(raw_items):
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            discard_counter["missing_question"] += 1
            discarded_indices.append(index)
            continue

        try:
            public_inputs = _coerce_string_list(item.get("example_input"), "example_input")
            public_outputs = _coerce_string_list(item.get("example_output"), "example_output")
            private_inputs = _coerce_string_list(item.get("test_input"), "test_input")
            private_outputs = _coerce_string_list(item.get("test_output"), "test_output")
        except (TypeError, ValueError):
            discard_counter["invalid_test_fields"] += 1
            discarded_indices.append(index)
            continue

        if len(public_inputs) != len(public_outputs):
            discard_counter["public_length_mismatch"] += 1
            discarded_indices.append(index)
            continue
        if len(private_inputs) != len(private_outputs):
            discard_counter["private_length_mismatch"] += 1
            discarded_indices.append(index)
            continue
        if len(public_inputs) == 0:
            discard_counter["no_public_examples"] += 1
            discarded_indices.append(index)
            continue
        if len(private_inputs) == 0:
            discard_counter["no_private_tests"] += 1
            discarded_indices.append(index)
            continue

        public_tests = [
            Test(input=test_input, output=test_output, testtype=TestType.STDIN)
            for test_input, test_output in zip(public_inputs, public_outputs, strict=True)
        ]
        public_source_counter["example_input/example_output"] += 1
        private_tests = [
            Test(input=test_input, output=test_output, testtype=TestType.STDIN)
            for test_input, test_output in zip(private_inputs, private_outputs, strict=True)
        ]

        question_content, embedded = _build_question_content(
            question,
            public_tests,
            embed_public_examples,
        )
        embedded_counter["embedded" if embedded else "kept_original"] += 1
        if mask_additional_sample_outputs:
            masked_question_content = hide_sample_outputs(question_content)
            mask_counter["masked" if masked_question_content != question_content else "unchanged"] += 1
            question_content = masked_question_content
        else:
            mask_counter["disabled"] += 1

        problem = _build_problem(
            dataset_name=dataset_name,
            item=item,
            index=index,
            question_content=question_content,
            public_tests=public_tests,
            private_tests=private_tests,
        )

        platform_counter[problem.platform.value] += 1
        difficulty_counter[problem.difficulty.value] += 1
        public_counts.append(len(public_tests))
        private_counts.append(len(private_tests))

        problem_json = codegen_problem_to_json_dict(problem)
        with (benchmark_dir / f"{problem.question_id}.json").open("w", encoding="utf-8") as f:
            json.dump(problem_json, f, ensure_ascii=False, indent=2)
        records.append(_build_generation_record(problem, data_source=config["data_source"]))

    if not records:
        raise ValueError(f"No valid problems found in {resolved_input}")

    pd.DataFrame.from_records(records).to_parquet(out_parquet, index=False)

    sanity_report = {
        "dataset": dataset_name,
        "input_json": str(resolved_input),
        "output_parquet": str(out_parquet),
        "benchmark_dir": str(benchmark_dir),
        "embed_public_examples": embed_public_examples,
        "mask_additional_sample_outputs": mask_additional_sample_outputs,
        "n_raw_items": len(raw_items),
        "n_kept": len(records),
        "n_discarded": len(raw_items) - len(records),
        "discard_reasons": dict(discard_counter),
        "discarded_indices": discarded_indices,
        "platform_counts": dict(platform_counter),
        "difficulty_counts": dict(difficulty_counter),
        "question_content_strategy": dict(embedded_counter),
        "sample_output_masking": dict(mask_counter),
        "public_test_source": dict(public_source_counter),
        "avg_public_test_count": sum(public_counts) / len(public_counts),
        "avg_private_test_count": sum(private_counts) / len(private_counts),
        "max_public_test_count": max(public_counts),
        "max_private_test_count": max(private_counts),
    }
    with sanity_path.open("w", encoding="utf-8") as f:
        json.dump(sanity_report, f, ensure_ascii=False, indent=2)

    print(
        f"[{dataset_name}] kept {len(records)}/{len(raw_items)} problems; "
        f"parquet -> {out_parquet}; benchmarks -> {benchmark_dir}; "
        f"report -> {sanity_path}"
    )
    return sanity_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=tuple(sorted(_DATASET_CONFIG.keys())),
        required=True,
        help="Which CURE evaluation benchmark to prepare.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional path to a local CURE JSON file. If omitted, download from Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to eval/<dataset>/shared.",
    )
    parser.add_argument(
        "--embed-public-examples",
        choices=("auto", "never", "always"),
        default="auto",
        help=(
            "How to handle example_input/example_output in question_content. "
            "'auto' appends them only when they do not already appear in the question."
        ),
    )
    parser.add_argument(
        "--disable-mask-additional-sample-outputs",
        action="store_true",
        help=(
            "By default, sample outputs beyond the first visible example are hidden "
            "to align with the LCB/TACO training protocol."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    output_dir = args.output_dir or _DATASET_CONFIG[args.dataset]["default_output_dir"]
    prepare_cure_code_benchmark(
        dataset_name=args.dataset,
        input_json=args.input_json,
        output_dir=output_dir,
        embed_public_examples=args.embed_public_examples,
        mask_additional_sample_outputs=not args.disable_mask_additional_sample_outputs,
        force=args.force,
    )


if __name__ == "__main__":
    main()
