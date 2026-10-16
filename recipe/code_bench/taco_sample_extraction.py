"""TACO question-markdown Sample I/O extraction and masking.

TACO aggregates problems from multiple competitive-programming platforms and
each platform writes Sample I/O blocks with a slightly different markdown
convention. Inspection of the TCS-filtered TACO_Train shows four concrete
formats in use:

  Format A (AtCoder new, CodeChef):
      -----Sample Input [N]?-----
      <content>
      -----Sample Output [N]?-----
      <content>

  Format B (Codeforces with dashes):
      -----Examples-----
      Input
      <content>

      Output
      <content>
      (repeated)

  Format C (Aizu, AtCoder old):
      Examples

      Input

      <content>


      Output

      <content>
      (repeated)

  Format D (Codeforces bare):
      Examples
      Input
      <content>

      Output
      <content>
      (repeated)

`extract_samples` tries each format in order and returns the first parse that
yields ≥ 1 sample. `mask_markdown_sample_outputs` mirrors the format-detection
logic and hides Sample Output 2+ content in-place.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Format A: `-----Sample Input [N]?-----` / `-----Sample Output [N]?-----`
# ---------------------------------------------------------------------------

_FMT_A_BLOCK_RE = re.compile(
    r"-----+\s*Sample\s*Input\s*(\d*)\s*-----+\s*\n(.*?)\n"
    r"-----+\s*Sample\s*Output\s*\1\s*-----+\s*\n(.*?)"
    # Stop at any `-----` delimiter so trailing `----- explanation N ------`
    # (common in CodeChef) doesn't bleed into Sample Output's content. The
    # follow-up sample block (`-----Sample Input 2-----`) still matches this
    # stop because it starts with `-----`.
    r"(?=\n-----+|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_fmt_a(markdown: str) -> list[dict[str, str]]:
    return [
        {"input": m.group(2).rstrip("\n"), "output": m.group(3).rstrip("\n")}
        for m in _FMT_A_BLOCK_RE.finditer(markdown)
    ]


# ---------------------------------------------------------------------------
# Formats B & D: Examples section with `Input\n<content>\n\nOutput\n<content>`
# (B uses `-----Examples-----`, D uses bare `Examples`.) Same inner parser.
# ---------------------------------------------------------------------------

_FMT_B_SECTION_RE = re.compile(
    r"-----+\s*Examples?\s*-----+\s*\n(.*?)(?=\n-----+|\Z)", re.DOTALL | re.IGNORECASE
)
_FMT_D_SECTION_RE = re.compile(
    # Bare `Examples` header not preceded by dashes. Stop at next section-like
    # marker (Note / Constraints / Explanation) or end of markdown.
    r"(?<!-)\n\s*Examples?\s*\n(.*?)(?=\n\s*(?:Note|Constraints|Explanation|-----)\s*\n|\Z)",
    re.DOTALL,
)
_BD_BLOCK_RE = re.compile(
    # Input / Output labels with varying blank-line separation. Content stops
    # at the next `Input` label or end of section.
    #
    # `Output\s*?\n` is non-greedy on purpose: when a sample has an EMPTY
    # output (e.g. AtCoder problems printing nothing for trivial inputs), a
    # greedy `Output\s*\n` would swallow every newline between the empty
    # output and the next `Input` label, leaving the following `(.*?)` with
    # no way to recognise the block boundary — it then expands through the
    # rest of the section, folding 2+ samples into one match.
    # The captured `(.*?)` groups may still carry leading/trailing newlines,
    # which the caller strips (see `_extract_from_section` below).
    r"Input\s*?\n(.*?)\n\s*Output\s*?\n(.*?)(?=\s*Input\s*\n|\Z)",
    re.DOTALL,
)


def _extract_from_section(section: str) -> list[dict[str, str]]:
    return [
        {"input": m.group(1).strip("\n"), "output": m.group(2).strip("\n")}
        for m in _BD_BLOCK_RE.finditer(section)
    ]


def _extract_fmt_b(markdown: str) -> list[dict[str, str]]:
    section = _FMT_B_SECTION_RE.search(markdown)
    if not section:
        return []
    return _extract_from_section(section.group(1))


def _extract_fmt_d(markdown: str) -> list[dict[str, str]]:
    section = _FMT_D_SECTION_RE.search(markdown)
    if not section:
        return []
    return _extract_from_section(section.group(1))


# ---------------------------------------------------------------------------
# Format C: `Examples` + `Input\n\n<content>\n\n\nOutput\n\n<content>` (aizu, old atcoder)
# ---------------------------------------------------------------------------

_FMT_C_BLOCK_RE = re.compile(
    r"Input\s*\n+(.*?)\n+\s*Output\s*\n+(.*?)(?=\n+\s*Input\s*\n|\Z)",
    re.DOTALL,
)


def _extract_fmt_c(markdown: str) -> list[dict[str, str]]:
    section = _FMT_D_SECTION_RE.search(markdown)
    if not section:
        return []
    return [
        {"input": m.group(1).rstrip("\n"), "output": m.group(2).rstrip("\n")}
        for m in _FMT_C_BLOCK_RE.finditer(section.group(1))
    ]


# ---------------------------------------------------------------------------
# Format E: HackerEarth / HackerRank style `Sample Input N?` + `Sample Output N?`
# (labels spelled out, no `-----` frame, N is optional)
# ---------------------------------------------------------------------------

_FMT_E_BLOCK_RE = re.compile(
    r"(?:SAMPLE|Sample)\s+(?:INPUT|Input)\s*(\d*)\s*\n(.*?)\n"
    r"\s*(?:SAMPLE|Sample)\s+(?:OUTPUT|Output)\s*\1\s*\n(.*?)"
    # End content on the NEXT sample block (numbered or unnumbered) or on a
    # recognized section header. The `\s*\d*\s*` segment after `Input` is what
    # lets numbered follow-up blocks like `Sample Input 1` / `Sample Input 2`
    # terminate the current match; without it the lookahead only catches
    # unnumbered follow-ups and the first block's output absorbs the rest.
    r"(?=\n\s*(?:SAMPLE|Sample)\s+(?:INPUT|Input)\s*\d*\s*\n|\n\s*(?:Explanation|Constraints|Note)\b|\Z)",
    re.DOTALL,
)


def _extract_fmt_e(markdown: str) -> list[dict[str, str]]:
    return [
        {"input": m.group(2).rstrip("\n"), "output": m.group(3).rstrip("\n")}
        for m in _FMT_E_BLOCK_RE.finditer(markdown)
    ]


# ---------------------------------------------------------------------------
# Extractor dispatch: try formats in order, return first that yields ≥ 1 sample.
# ---------------------------------------------------------------------------


_EXTRACTORS = (
    ("A", _extract_fmt_a),
    ("B", _extract_fmt_b),
    ("D", _extract_fmt_d),
    ("C", _extract_fmt_c),
    ("E", _extract_fmt_e),
)


def detect_format(markdown: str) -> str | None:
    """Return the first format tag ('A' / 'B' / 'D' / 'C') that yields ≥ 1 sample,
    or None if no extractor matches."""
    for tag, extractor in _EXTRACTORS:
        if extractor(markdown):
            return tag
    return None


def extract_samples(markdown: str, source: str | None = None) -> list[dict[str, str]]:
    """Parse Sample I/O blocks from a TACO problem markdown.

    Returns a list of {input, output} dicts. An empty list means no recognized
    format was found — the caller should discard the problem rather than fall
    back to the hidden `inputs[0:N]` array (TACO's `input_output` does NOT
    always start with the markdown samples, notably for aizu/codechef).

    `source` is accepted for API symmetry but not used: we detect the format
    directly from markdown so the same problem can parse regardless of which
    platform string TACO labels it with.
    """
    del source
    for _tag, extractor in _EXTRACTORS:
        samples = extractor(markdown)
        if samples:
            return samples
    return []


# ---------------------------------------------------------------------------
# Masking: hide Sample Output 2+ content. Keep Sample Input blocks verbatim so
# the model still sees the input format for all samples.
# ---------------------------------------------------------------------------

_HIDDEN_TOKEN = "[hidden]"


def _mask_fmt_a(markdown: str) -> tuple[str, int]:
    blocks = list(_FMT_A_BLOCK_RE.finditer(markdown))
    if len(blocks) < 2:
        return markdown, 0

    def _strip_following_explanation(gap: str, num_str: str) -> str:
        num_pattern = re.escape(num_str) if num_str else r"\d*"
        return re.sub(
            rf"^\s*-----+\s*explanation\s*{num_pattern}\s*-----+.*\Z",
            "",
            gap,
            flags=re.IGNORECASE | re.DOTALL,
        )

    parts: list[str] = [markdown[: blocks[0].start()]]
    n_masked = 0
    for idx, m in enumerate(blocks):
        num_str = m.group(1)
        input_content = m.group(2).rstrip("\n")
        output_content = m.group(3).rstrip("\n") if idx == 0 else _HIDDEN_TOKEN
        n_masked += 0 if idx == 0 else 1
        num_label = f" {num_str}" if num_str else ""
        parts.append(
            f"-----Sample Input{num_label}-----\n{input_content}\n"
            f"-----Sample Output{num_label}-----\n{output_content}"
        )
        gap_start = m.end()
        gap_end = blocks[idx + 1].start() if idx + 1 < len(blocks) else len(markdown)
        gap = markdown[gap_start:gap_end]
        if idx >= 1:
            gap = _strip_following_explanation(gap, num_str)
        parts.append(gap)
    return "".join(parts), n_masked


def _mask_section_bd(section: str, block_re: re.Pattern) -> tuple[str, int]:
    blocks = list(block_re.finditer(section))
    if len(blocks) < 2:
        return section, 0
    parts: list[str] = []
    cursor = 0
    n_masked = 0
    for idx, m in enumerate(blocks):
        parts.append(section[cursor : m.start()])
        input_content = m.group(1).strip("\n")
        output_content = m.group(2).strip("\n") if idx == 0 else _HIDDEN_TOKEN
        n_masked += 0 if idx == 0 else 1
        parts.append(f"Input\n{input_content}\n\nOutput\n{output_content}")
        cursor = m.end()
    parts.append(section[cursor:])
    return "".join(parts), n_masked


def _mask_fmt_b(markdown: str) -> tuple[str, int]:
    section_match = _FMT_B_SECTION_RE.search(markdown)
    if not section_match:
        return markdown, 0
    masked_section, n_masked = _mask_section_bd(section_match.group(1), _BD_BLOCK_RE)
    if n_masked == 0:
        return markdown, 0
    return (
        markdown[: section_match.start(1)] + masked_section + markdown[section_match.end(1) :],
        n_masked,
    )


def _mask_fmt_d(markdown: str) -> tuple[str, int]:
    section_match = _FMT_D_SECTION_RE.search(markdown)
    if not section_match:
        return markdown, 0
    masked_section, n_masked = _mask_section_bd(section_match.group(1), _BD_BLOCK_RE)
    if n_masked == 0:
        return markdown, 0
    return (
        markdown[: section_match.start(1)] + masked_section + markdown[section_match.end(1) :],
        n_masked,
    )


def _mask_fmt_c(markdown: str) -> tuple[str, int]:
    section_match = _FMT_D_SECTION_RE.search(markdown)
    if not section_match:
        return markdown, 0
    section = section_match.group(1)
    blocks = list(_FMT_C_BLOCK_RE.finditer(section))
    if len(blocks) < 2:
        return markdown, 0
    parts: list[str] = []
    cursor = 0
    n_masked = 0
    for idx, m in enumerate(blocks):
        parts.append(section[cursor : m.start()])
        input_content = m.group(1).rstrip("\n")
        output_content = m.group(2).rstrip("\n") if idx == 0 else _HIDDEN_TOKEN
        n_masked += 0 if idx == 0 else 1
        parts.append(f"Input\n\n{input_content}\n\n\nOutput\n\n{output_content}")
        cursor = m.end()
    parts.append(section[cursor:])
    masked_section = "".join(parts)
    if n_masked == 0:
        return markdown, 0
    return (
        markdown[: section_match.start(1)] + masked_section + markdown[section_match.end(1) :],
        n_masked,
    )


def _mask_fmt_e(markdown: str) -> tuple[str, int]:
    blocks = list(_FMT_E_BLOCK_RE.finditer(markdown))
    if len(blocks) < 2:
        return markdown, 0
    parts: list[str] = []
    cursor = 0
    n_masked = 0
    for idx, m in enumerate(blocks):
        parts.append(markdown[cursor : m.start()])
        num_str = m.group(1)
        input_content = m.group(2).rstrip("\n")
        output_content = m.group(3).rstrip("\n") if idx == 0 else _HIDDEN_TOKEN
        n_masked += 0 if idx == 0 else 1
        num_label = f" {num_str}" if num_str else ""
        parts.append(
            f"Sample Input{num_label}\n{input_content}\n"
            f"Sample Output{num_label}\n{output_content}"
        )
        cursor = m.end()
    parts.append(markdown[cursor:])
    return "".join(parts), n_masked


_MASKERS = (
    ("A", _mask_fmt_a),
    ("B", _mask_fmt_b),
    ("D", _mask_fmt_d),
    ("C", _mask_fmt_c),
    ("E", _mask_fmt_e),
)


def mask_markdown_sample_outputs(markdown: str, source: str | None = None) -> str:
    """Hide Sample Output 2+ content while preserving all Sample Inputs.

    We try the same format detectors as `extract_samples` and use the first
    matcher that actually finds ≥ 2 sample blocks (single-sample problems have
    nothing to mask). Unmatched markdown is returned verbatim.
    """
    del source
    for _tag, masker in _MASKERS:
        masked, n_masked = masker(markdown)
        if n_masked > 0:
            return masked
    return markdown


# ---------------------------------------------------------------------------
# Dedup: remove private tests that duplicate a public test input.
# ---------------------------------------------------------------------------


def _normalize_for_compare(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.strip().replace("\r\n", "\n")


def remove_public_overlap(
    public_tests: list[dict[str, str]],
    private_inputs: list[str],
    private_outputs: list[str],
) -> tuple[list[str], list[str], int]:
    """Drop private tests whose input matches any public test input.

    Keeps ordering of private_inputs/outputs. Returns `(kept_inputs,
    kept_outputs, n_overlap)`.
    """
    if len(private_inputs) != len(private_outputs):
        raise ValueError(
            f"private inputs/outputs length mismatch: {len(private_inputs)} vs {len(private_outputs)}"
        )
    public_keys = {_normalize_for_compare(sample["input"]) for sample in public_tests}
    keep_inputs: list[str] = []
    keep_outputs: list[str] = []
    overlap = 0
    for inp, out in zip(private_inputs, private_outputs, strict=True):
        if _normalize_for_compare(inp) in public_keys:
            overlap += 1
            continue
        keep_inputs.append(inp)
        keep_outputs.append(out)
    return keep_inputs, keep_outputs, overlap
