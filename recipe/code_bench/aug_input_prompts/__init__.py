"""Versioned prompts for iterative probe (aug-input) generation.

A version is a `v<N>.py` module that exports `SYSTEM` (str) and
`USER_TEMPLATE` (str), registered in `PROMPT_REGISTRY` below. Shipped versions
are immutable so generated parquets stay reproducible from their stamped
`prompt_version` column.

Placeholders required in `USER_TEMPLATE`:
  {iter_idx} {k_total} {public_inputs_json} {prior_inputs_json} {problem_content}
"""
from . import v1

PROMPT_REGISTRY = {
    "v1": v1,
}


def get_prompt(version: str):
    if version not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt version: {version!r}. "
            f"Available: {sorted(PROMPT_REGISTRY)}"
        )
    return PROMPT_REGISTRY[version]
