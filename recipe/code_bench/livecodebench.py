"""
Shared LiveCodeBench v6 helpers for recipe-local preparation and evaluation.

Use this module when normalizing raw LiveCodeBench records, writing per-question
benchmark JSON files, building generation prompts, extracting code from model
responses, or converting a benchmark problem into runnable test cases.
"""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

REQUIRED_RAW_FIELDS = {
    "question_title",
    "question_content",
    "platform",
    "question_id",
    "contest_id",
    "contest_date",
    "starter_code",
    "difficulty",
    "public_test_cases",
    "private_test_cases",
    "metadata",
}

SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads "
    "the inputs, runs the algorithm and writes output to STDOUT."
)


class Platform(str, Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"
    # TACO-native platforms (reused by prepare_taco); LCB data never emits these.
    CODECHEF = "codechef"
    HACKEREARTH = "hackerearth"
    CODEWARS = "codewars"
    AIZU = "aizu"
    GEEKSFORGEEKS = "geeksforgeeks"
    TACO_OTHER = "taco-other"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    # TACO-native difficulty tiers.
    MEDIUM_HARD = "medium_hard"
    VERY_HARD = "very_hard"
    UNKNOWN_DIFFICULTY = "unknown_difficulty"


class TestType(str, Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass(frozen=True)
class Test:
    input: str
    output: str
    testtype: TestType


@dataclass(frozen=True)
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: tuple[Test, ...]
    private_test_cases: tuple[Test, ...]
    metadata: dict[str, Any]
    ground_truth_solution: str | None = None


def _validate_required_fields(raw_problem: dict[str, Any], line_number: int | None = None) -> None:
    missing_fields = REQUIRED_RAW_FIELDS - raw_problem.keys()
    if missing_fields:
        prefix = f"Line {line_number} " if line_number is not None else ""
        raise KeyError(f"{prefix}is missing required fields: {sorted(missing_fields)}")


def _parse_test_case_list(raw_value: Any, field_name: str) -> list[dict[str, Any]]:
    parsed_value: Any
    if isinstance(raw_value, list):
        parsed_value = raw_value
    elif isinstance(raw_value, str):
        parsed_value = json.loads(raw_value)
    else:
        raise TypeError(f"{field_name} must be a JSON string or list, got {type(raw_value)}")

    if not isinstance(parsed_value, list):
        raise TypeError(f"{field_name} must decode to a list, got {type(parsed_value)}")
    return parsed_value


def _parse_private_test_case_list(raw_value: Any) -> list[dict[str, Any]]:
    if isinstance(raw_value, list):
        parsed_value = raw_value
    elif isinstance(raw_value, str):
        stripped_value = raw_value.strip()
        if stripped_value.startswith("["):
            parsed_value = json.loads(stripped_value)
        else:
            decoded_value = pickle.loads(zlib.decompress(base64.b64decode(raw_value.encode("utf-8"))))
            if isinstance(decoded_value, str):
                parsed_value = json.loads(decoded_value)
            elif isinstance(decoded_value, list):
                parsed_value = decoded_value
            else:
                raise TypeError(
                    "Decoded private_test_cases must be a JSON string or list, "
                    f"got {type(decoded_value)}"
                )
    else:
        raise TypeError(f"private_test_cases must be a JSON string or list, got {type(raw_value)}")

    if not isinstance(parsed_value, list):
        raise TypeError(f"private_test_cases must decode to a list, got {type(parsed_value)}")
    return parsed_value


def _parse_metadata(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        metadata = raw_value
    elif isinstance(raw_value, str):
        metadata = json.loads(raw_value)
    else:
        raise TypeError(f"metadata must be a JSON string or dict, got {type(raw_value)}")

    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must decode to a dict, got {type(metadata)}")
    return metadata


def _parse_optional_string(raw_value: Any, field_name: str) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise TypeError(f"{field_name} must be a string or null, got {type(raw_value)}")

    if raw_value.strip() == "":
        raise ValueError(f"{field_name} must not be empty when provided")
    return raw_value


def _parse_test(raw_test: dict[str, Any], field_name: str, idx: int) -> Test:
    required_fields = {"input", "output", "testtype"}
    missing_fields = required_fields - raw_test.keys()
    if missing_fields:
        raise KeyError(f"{field_name}[{idx}] is missing required fields: {sorted(missing_fields)}")

    input_value = raw_test["input"]
    output_value = raw_test["output"]
    testtype_value = raw_test["testtype"]

    if not isinstance(input_value, str):
        raise TypeError(f"{field_name}[{idx}].input must be a string, got {type(input_value)}")
    if not isinstance(output_value, str):
        raise TypeError(f"{field_name}[{idx}].output must be a string, got {type(output_value)}")

    return Test(input=input_value, output=output_value, testtype=TestType(testtype_value))


def codegen_problem_from_json_dict(raw_problem: dict[str, Any], line_number: int | None = None) -> CodeGenerationProblem:
    _validate_required_fields(raw_problem, line_number)

    public_test_cases = tuple(
        _parse_test(raw_test, "public_test_cases", idx)
        for idx, raw_test in enumerate(_parse_test_case_list(raw_problem["public_test_cases"], "public_test_cases"))
    )
    private_test_cases = tuple(
        _parse_test(raw_test, "private_test_cases", idx)
        for idx, raw_test in enumerate(_parse_private_test_case_list(raw_problem["private_test_cases"]))
    )

    question_title = raw_problem["question_title"]
    question_content = raw_problem["question_content"]
    question_id = raw_problem["question_id"]
    contest_id = raw_problem["contest_id"]
    contest_date = raw_problem["contest_date"]
    starter_code = raw_problem["starter_code"]

    if not isinstance(question_title, str):
        raise TypeError(f"question_title must be a string, got {type(question_title)}")
    if not isinstance(question_content, str):
        raise TypeError(f"question_content must be a string, got {type(question_content)}")
    if not isinstance(question_id, str):
        raise TypeError(f"question_id must be a string, got {type(question_id)}")
    if not isinstance(contest_id, str):
        raise TypeError(f"contest_id must be a string, got {type(contest_id)}")
    if not isinstance(contest_date, str):
        raise TypeError(f"contest_date must be a string, got {type(contest_date)}")
    if not isinstance(starter_code, str):
        raise TypeError(f"starter_code must be a string, got {type(starter_code)}")

    return CodeGenerationProblem(
        question_title=question_title,
        question_content=question_content,
        platform=Platform(raw_problem["platform"]),
        question_id=question_id,
        contest_id=contest_id,
        contest_date=datetime.fromisoformat(contest_date),
        starter_code=starter_code,
        difficulty=Difficulty(raw_problem["difficulty"]),
        public_test_cases=public_test_cases,
        private_test_cases=private_test_cases,
        metadata=_parse_metadata(raw_problem["metadata"]),
        ground_truth_solution=_parse_optional_string(raw_problem.get("ground_truth_solution"), "ground_truth_solution"),
    )


def _test_to_json_dict(test: Test) -> dict[str, str]:
    return {"input": test.input, "output": test.output, "testtype": test.testtype.value}


def codegen_problem_to_json_dict(problem: CodeGenerationProblem) -> dict[str, Any]:
    problem_json = {
        "question_title": problem.question_title,
        "question_content": problem.question_content,
        "platform": problem.platform.value,
        "question_id": problem.question_id,
        "contest_id": problem.contest_id,
        "contest_date": problem.contest_date.isoformat(),
        "starter_code": problem.starter_code,
        "difficulty": problem.difficulty.value,
        "public_test_cases": [_test_to_json_dict(test) for test in problem.public_test_cases],
        "private_test_cases": [_test_to_json_dict(test) for test in problem.private_test_cases],
        "metadata": problem.metadata,
    }
    if problem.ground_truth_solution is not None:
        problem_json["ground_truth_solution"] = problem.ground_truth_solution
    return problem_json


def load_codegen_problem(json_path: str | Path) -> CodeGenerationProblem:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return codegen_problem_from_json_dict(json.load(f))


def derive_public_private_pass_flags(
    problem: CodeGenerationProblem,
    results: list[Any],
) -> tuple[bool, bool]:
    """Derive public/private pass flags from prefix evaluator results.

    LiveCodeBench evaluation stops early on the first failing test, so `results`
    is a prefix rather than a full-length test array. Public tests are always
    executed first, and private tests are only reached if all public tests pass.
    """

    public_test_count = len(problem.public_test_cases)
    private_test_count = len(problem.private_test_cases)
    total_test_count = public_test_count + private_test_count
    if len(results) > total_test_count:
        raise ValueError(
            "results length mismatch for question_id="
            f"{problem.question_id}: expected at most {total_test_count} results "
            f"({public_test_count} public + {private_test_count} private), got {len(results)}"
        )

    public_results = results[:public_test_count]
    public_pass = public_test_count == 0 or (
        len(public_results) == public_test_count and all(item is True for item in public_results)
    )
    if not public_pass:
        return False, False

    if private_test_count == 0:
        return True, True

    private_results = results[public_test_count:]
    private_pass = len(results) == total_test_count and all(item is True for item in private_results)
    return True, private_pass


def split_results_by_test_split(
    problem: CodeGenerationProblem,
    results: list[Any],
) -> tuple[bool, bool]:
    """Backward-compatible alias for prefix-aware public/private pass flags."""

    return derive_public_private_pass_flags(problem, results)


def build_user_prompt(problem: CodeGenerationProblem) -> str:
    prompt = f"### Question:\n{problem.question_content}\n\n"
    if problem.starter_code:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def format_prompt_generation(problem: CodeGenerationProblem) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": build_user_prompt(problem)},
    ]


def _build_in_outs_from_tests(problem: CodeGenerationProblem, tests: list[Test] | tuple[Test, ...]) -> dict[str, Any]:
    return {
        "inputs": [test.input for test in tests],
        "outputs": [test.output for test in tests],
        "fn_name": problem.metadata.get("func_name"),
    }


def build_public_in_outs(problem: CodeGenerationProblem) -> dict[str, Any]:
    return _build_in_outs_from_tests(problem, problem.public_test_cases)


def build_private_in_outs(problem: CodeGenerationProblem) -> dict[str, Any]:
    return _build_in_outs_from_tests(problem, problem.private_test_cases)


def build_in_outs(problem: CodeGenerationProblem) -> dict[str, Any]:
    all_test_cases = [*problem.public_test_cases, *problem.private_test_cases]
    return _build_in_outs_from_tests(problem, all_test_cases)


def extract_code_from_response(model_output: str) -> str:
    if not isinstance(model_output, str):
        raise TypeError(f"model_output must be a string, got {type(model_output)}")

    candidate = model_output
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[1]

    output_lines = candidate.split("\n")
    fence_indices = [idx for idx, line in enumerate(output_lines) if "```" in line]
    if len(fence_indices) < 2:
        return ""
    return "\n".join(output_lines[fence_indices[-2] + 1 : fence_indices[-1]]).strip()
