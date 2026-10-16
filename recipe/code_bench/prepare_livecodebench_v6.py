"""
Prepare LiveCodeBench v6 raw JSONL into generation parquet and benchmark JSON files.

Use this script when converting the raw benchmark dump into the normalized
recipe-local format consumed by standalone generation and offline pass@k
evaluation.

Example:
    python -m recipe.code_bench.prepare_livecodebench_v6 \
        --raw-jsonl /path/to/test6.jsonl \
        --output-parquet /path/to/livecodebench_v6.parquet \
        --benchmark-dir /path/to/benchmark_json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from recipe.code_bench.livecodebench import (
    codegen_problem_from_json_dict,
    codegen_problem_to_json_dict,
    format_prompt_generation,
)


def build_generation_record(problem_json: dict) -> dict:
    problem = codegen_problem_from_json_dict(problem_json)
    return {
        "data_source": "livecodebench_v6",
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


def prepare_livecodebench_v6(raw_jsonl: Path, output_parquet: Path, benchmark_dir: Path) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen_question_ids: set[str] = set()

    with raw_jsonl.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                raise ValueError(f"Line {line_number} is empty")

            raw_problem = json.loads(line)
            problem = codegen_problem_from_json_dict(raw_problem, line_number=line_number)

            if problem.question_id in seen_question_ids:
                raise ValueError(f"Duplicate question_id detected: {problem.question_id}")
            seen_question_ids.add(problem.question_id)

            benchmark_path = benchmark_dir / f"{problem.question_id}.json"
            if benchmark_path.exists():
                with benchmark_path.open("r", encoding="utf-8") as benchmark_file:
                    existing_problem = codegen_problem_from_json_dict(json.load(benchmark_file))
                if existing_problem.ground_truth_solution is not None:
                    problem = replace(problem, ground_truth_solution=existing_problem.ground_truth_solution)

            problem_json = codegen_problem_to_json_dict(problem)
            with benchmark_path.open("w", encoding="utf-8") as benchmark_file:
                json.dump(problem_json, benchmark_file, ensure_ascii=False, indent=2)

            records.append(build_generation_record(problem_json))

    pd.DataFrame.from_records(records).to_parquet(output_parquet, index=False)
    print(
        f"Prepared {len(records)} LiveCodeBench v6 problems into "
        f"{output_parquet} and {benchmark_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    args = parser.parse_args()

    prepare_livecodebench_v6(
        raw_jsonl=args.raw_jsonl,
        output_parquet=args.output_parquet,
        benchmark_dir=args.benchmark_dir,
    )


if __name__ == "__main__":
    main()
