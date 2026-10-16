"""
Create a training parquet with hidden Sample Outputs for samples 2+.

For each question's prompt, keep the first Sample I/O complete (for format
understanding), but remove the expected output for all subsequent samples.
This prevents the model from hardcoding Sample I/O to cheat public tests.

Handles multiple platform formats:
  - AtCoder / many CP statements: "Sample Input N" / "Sample Output N"
  - Codeforces (CURE format): "Example Input N:" / "Example Output N:"
  - Uppercase sample blocks: "SAMPLE INPUT" / "SAMPLE OUTPUT"
  - Generic "Examples" sections with repeated "Input" / "Output" pairs
  - LeetCode: "Example N:" with "Input:" / "Output:" / "Explanation:"

Usage:
    python -m recipe.code_bench.create_hidden_output_parquet \
        --input-parquet eval/livecodebench_v6/shared/test6.parquet \
        --output-parquet eval/livecodebench_v6/shared/test6_hidden_output.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _hide_atcoder_outputs(text: str) -> str:
    """Remove Sample Output N (N>=2) and any trailing explanation."""
    # Match: "Sample Output N\n\n<content>" up to the next "Sample Input" or end-of-samples
    # We keep "Sample Input N\n\n<input>" intact.
    def _replace(m: re.Match) -> str:
        return ""

    # Remove "Sample Output N" blocks for N >= 2.
    # Pattern: "Sample Output N\n\n" followed by content until next "Sample Input" or "### Format" or end.
    result = re.sub(
        r"Sample Output ([2-9]|\d{2,})\n\n.*?(?=Sample Input \d+|### Format|\Z)",
        _replace,
        text,
        flags=re.DOTALL,
    )
    return result


def _hide_codeforces_outputs(text: str) -> str:
    """Remove Example Output N (all but the first occurrence)."""
    matches = list(re.finditer(r"Example Output \d+:", text, flags=re.IGNORECASE))
    if len(matches) <= 1:
        return text

    kept_first = False
    result_parts: list[str] = []
    cursor = 0
    for match in matches:
        if not kept_first:
            kept_first = True
            continue

        result_parts.append(text[cursor : match.start()])
        tail = text[match.end() :]
        next_header = re.search(
            r"(?=Example Input \d+:|Example Output \d+:|Notes?:|Explanation:|\Z)",
            tail,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if next_header is None:
            cursor = len(text)
        else:
            cursor = match.end() + next_header.start()

    result_parts.append(text[cursor:])
    return "".join(result_parts)


def _hide_generic_sample_outputs(
    text: str,
    *,
    output_header_pattern: str,
    next_header_pattern: str,
    case_insensitive: bool = True,
) -> str:
    """Remove all but the first sample-output block for a header family.

    Set ``case_insensitive=False`` when the pattern is an all-uppercase variant
    (e.g. ``SAMPLE OUTPUT``) that must not match mixed-case AtCoder headers —
    otherwise the header regex over-matches and strips too much.
    """
    base_flags = re.MULTILINE | (re.IGNORECASE if case_insensitive else 0)
    matches = list(re.finditer(output_header_pattern, text, flags=base_flags))
    if len(matches) <= 1:
        return text

    result_parts: list[str] = []
    cursor = 0
    for idx, match in enumerate(matches):
        if idx == 0:
            continue
        result_parts.append(text[cursor : match.start()])
        tail = text[match.end() :]
        next_header = re.search(
            next_header_pattern,
            tail,
            flags=base_flags | re.DOTALL,
        )
        if next_header is None:
            cursor = len(text)
        else:
            cursor = match.end() + next_header.start()
    result_parts.append(text[cursor:])
    return "".join(result_parts)


def _hide_examples_section_outputs(text: str) -> str:
    """Within an Examples section, remove Output blocks after the first pair."""
    header_match = re.search(r"(^|\n)Examples?\s*(\n|$)", text, flags=re.IGNORECASE)
    if header_match is None:
        return text

    prefix = text[: header_match.end()]
    section = text[header_match.end() :]
    output_matches = list(re.finditer(r"(?im)^\s*Output\s*$", section))
    if len(output_matches) <= 1:
        return text

    result_parts: list[str] = []
    cursor = 0
    for idx, match in enumerate(output_matches):
        if idx == 0:
            continue
        result_parts.append(section[cursor : match.start()])
        tail = section[match.end() :]
        next_header = re.search(
            r"(?im)(?=^\s*Input\s*$|^\s*Examples?\s*$|^\s*Notes?\s*:?\s*$|^\s*Constraints?\s*:?\s*$|^\s*Scoring\s*:?\s*$|\Z)",
            tail,
        )
        if next_header is None:
            cursor = len(section)
        else:
            cursor = match.end() + next_header.start()
    result_parts.append(section[cursor:])
    return prefix + "".join(result_parts)


_EXAMPLE_HEADER_RE = re.compile(r"(?m)^Example (\d+):?[ \t]*$")


def _hide_leetcode_outputs(text: str) -> str:
    """Remove Output/Explanation from Example N (N>=2).

    Recognizes both ``Example N:`` (LeetCode-style, with colon) and
    ``Example N`` (CodeContests-style, no colon) headers, but only when each
    appears on its own line. Inline references like ``Example 2: ...`` inside
    a paragraph (e.g. the explanation section of CC problems) are not treated
    as headers, so masking lands on real test-case sections rather than
    paragraph prose.
    """
    headers = list(_EXAMPLE_HEADER_RE.finditer(text))
    if not headers:
        return text

    chunks: list[str] = [text[: headers[0].start()]]
    for i, m in enumerate(headers):
        n = int(m.group(1))
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[m.start(): block_end]
        if n >= 2:
            # The chunk boundary already excludes the next Example header, so
            # the lookahead only needs intra-block stop markers.
            chunk = re.sub(
                r"Output:.*?(?=\nInput:|\nConstraints:|\n\n\s*\n|\Z)",
                "",
                chunk,
                flags=re.DOTALL,
            )
            chunk = re.sub(
                r"Explanation:.*?(?=\nInput:|\nConstraints:|\n\n\s*\n|\Z)",
                "",
                chunk,
                flags=re.DOTALL,
            )
        chunks.append(chunk)
    return "".join(chunks)


def hide_sample_outputs(user_content: str) -> str:
    """Hide outputs for sample 2+ in the user prompt content."""
    result = user_content

    if re.search(r"Sample Input\s*1", result, flags=re.IGNORECASE):
        result = _hide_atcoder_outputs(result)

    if re.search(r"Example Output\s*\d+\s*:", result, flags=re.IGNORECASE):
        result = _hide_codeforces_outputs(result)

    if len(re.findall(r"Sample Output\s*:", result, flags=re.IGNORECASE)) > 1:
        result = _hide_generic_sample_outputs(
            result,
            output_header_pattern=r"Sample Output\s*:",
            next_header_pattern=r"(?=Sample Input\s*:|Sample Output\s*:|Notes?:|Explanation:|\Z)",
        )

    if len(re.findall(r"SAMPLE OUTPUT", result)) > 1:
        # Case-sensitive: this detector targets all-uppercase `SAMPLE OUTPUT`
        # variants (e.g. some codechef statements). Matching case-insensitively
        # would over-trigger on AtCoder's `Sample Output N` and strip large
        # sections via the `next_header_pattern`.
        result = _hide_generic_sample_outputs(
            result,
            output_header_pattern=r"SAMPLE OUTPUT",
            next_header_pattern=r"(?=SAMPLE INPUT|SAMPLE OUTPUT|Notes?:|Explanation:|\Z)",
            case_insensitive=False,
        )

    if re.search(r"(^|\n)Examples?\s*(\n|$)", result, flags=re.IGNORECASE):
        result = _hide_examples_section_outputs(result)

    if _EXAMPLE_HEADER_RE.search(result):
        result = _hide_leetcode_outputs(result)

    return result


def process_prompt(prompt_array) -> list[dict[str, str]]:
    """Process a prompt (list of message dicts) to hide sample outputs."""
    messages = list(prompt_array)
    result = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("role") == "user":
            msg["content"] = hide_sample_outputs(msg["content"])
        result.append(msg)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    df["prompt"] = df["prompt"].apply(process_prompt)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)
    print(f"Created {args.output_parquet} with hidden sample outputs ({len(df)} rows)")


if __name__ == "__main__":
    main()
