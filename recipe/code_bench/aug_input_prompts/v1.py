"""Probe (aug-input) prompt v1 — the prompt used for the paper's LiveCodeBench probes.

4-step verification pattern (Reasoning / Constraint check / Distinctness check /
Final Input), single input per call, prior_inputs seeded for diversification.
Hard size cap 5000 chars (conservative under max_tokens=10000-32000).

This file is immutable — to iterate, add a new version and bump --prompt-version.
"""

SYSTEM = (
    "You are an expert at designing test inputs for competitive programming problems. "
    "You generate test inputs that are strictly format-correct, satisfy all stated "
    "constraints, and are structurally distinct from any inputs already produced."
)


USER_TEMPLATE = """You are generating test input #{iter_idx} of {k_total} for the following competitive programming problem.

# Canonical input format (CRITICAL)
The test harness uses a SPECIFIC raw-string format that is often different from the "Input: nums = [...]" notation in human-readable Examples. Here are this problem's public test inputs in EXACT canonical format:

```json
{public_inputs_json}
```

Your generated input MUST follow the same format style as the JSON values above (same quoting style for strings, same use of newlines between args, same bracket nesting for lists, etc.).

# Inputs already generated for this problem (you MUST produce something DIFFERENT from EVERY one of these)

```json
{prior_inputs_json}
```

# Your task
Generate ONE more test input that is structurally DIFFERENT from:
- All example inputs shown in the problem statement
- All inputs in the "already generated" JSON above

Follow this 4-step pattern. Do NOT skip any step.

1. **Reasoning** — what aspect of the problem do you want to test that has NOT been adequately tested by the priors? What structural dimensions (size, range, sparsity, pattern, character set, distribution, adversarial structure, …) are unexplored? Be specific.

2. **Constraint check** — enumerate every constraint stated in the problem (size bounds, value bounds, structural invariants, format invariants) and verify your planned input satisfies each one.

3. **Distinctness verification** — DO NOT use vague phrases. For EACH prior input listed above (problem-statement examples + already-generated), write one concrete comparison line of the form:
       "Prior X: had A; mine: has B; A != B on dimension Y"
   where A and B are CONCRETE VALUES (not abstract claims). Choose dimension Y from: size, value range, sparsity, character set, pattern (sorted / reverse / random / alternating / repeated / adversarial), sign distribution, shape, special tokens, etc.

4. **Final Input** — output the input string in the canonical format inside a fenced code block.

# Hard constraint on input size
Your raw input string MUST NOT exceed 5000 characters. To exercise large-size cases, use representative sizes 100-1000 (not literal 10^5). Do NOT enumerate huge arrays; use a compact pattern that still stresses the algorithm.

# Output format (EXACTLY this)

## Reasoning
<your reasoning>

## Constraint check
<enumerate constraints + verify>

## Distinctness verification
- Prior <name>: had ...; mine: has ...; A != B on dimension ...
- Prior <name>: had ...; mine: has ...; A != B on dimension ...
- ... (one line per prior including problem-statement examples)

## Final Input
```
<the single input string in canonical format>
```

# Problem

{problem_content}

Begin generating input #{iter_idx} now."""
