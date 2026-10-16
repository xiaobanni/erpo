"""Iteratively generate K probe (aug test) inputs per problem, accumulating priors.

Each problem makes K sequential generation calls. Each call's prompt contains:
  - The full problem statement
  - The canonical-format JSON of public test inputs (format reference)
  - The JSON of inputs already generated in prior iterations of this problem
  - A 4-step pattern requiring the model to (a) reason, (b) verify constraints,
    (c) for EACH prior list "prior had A; mine has B; A!=B", (d) emit a single
    input in canonical format.

Across problems we run concurrently up to --concurrency at a time.

Requires a vLLM OpenAI-compatible server (see recipe/code_bench/scripts/generate_probes.sh,
which starts the server and runs the two generation passes used for the paper):

    vllm serve Qwen/Qwen3-4B --data-parallel-size 8 --port 8000 --max-model-len 16000

    python -m recipe.code_bench.generate_aug_inputs_iterative \\
        --benchmark-json-dir eval/livecodebench_v6/shared/test6_benchmark_json \\
        --output-parquet eval/livecodebench_v6/probes/lcb_probes_K10_pass1.parquet \\
        --k 10 --concurrency 32 --max-tokens 10000 --temperature 0.8 --enable-thinking --prompt-version v1

With --incremental-from, problems whose existing parquet row already has >= K
unique non-empty inputs are passed through unchanged; the others get
(K - existing) additional iterations appended.

Output parquet: one row per problem with `iter_inputs` (list of raw input
strings, consumed by the PCR reward), `iter_responses`, `iter_prompts`, and a
synthetic `responses` markdown that concatenates all iterations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp
import pandas as pd

from recipe.code_bench.aug_input_prompts import get_prompt

# Prompts live in `recipe/code_bench/aug_input_prompts/v<N>.py`. Select via
# CLI `--prompt-version` (default v1).


FINAL_INPUT_RE = re.compile(r"## Final Input\s*\n```(?:[a-zA-Z]+)?\s*\n(.*?)\n```", re.DOTALL)


def build_messages(prompt, problem: dict, prior_inputs: list[str], iter_idx: int, k_total: int) -> list[dict]:
    pub = {
        f"Public Input {i + 1}": tc["input"]
        for i, tc in enumerate(problem.get("public_test_cases", []))
    }
    pri = {f"Input {i + 1}": x for i, x in enumerate(prior_inputs)}
    return [
        {"role": "system", "content": prompt.SYSTEM},
        {"role": "user", "content": prompt.USER_TEMPLATE.format(
            problem_content=problem["question_content"],
            public_inputs_json=json.dumps(pub, indent=2, ensure_ascii=False) if pub else "{}",
            prior_inputs_json=json.dumps(pri, indent=2, ensure_ascii=False) if pri else "{}  (none yet — this is the first input)",
            iter_idx=iter_idx + 1,
            k_total=k_total,
        )},
    ]


def extract_final_input(response_text: str) -> str:
    """Return the content of the LAST '## Final Input' fenced block, or ''.

    Stores the model's raw text verbatim — parsing tolerance (literal ``\\n``
    vs real newline, etc.) is handled by the PCR executor, which tries
    multiple strategies and picks the first that produces a valid input.
    """
    matches = list(FINAL_INPUT_RE.finditer(response_text))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


async def call_model(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.8,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    # Thinking mode + 32K max_tokens + high concurrency can push individual
    # requests past 10 min; use 30 min so vLLM-side queueing doesn't trip aiohttp.
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as resp:
        result = await resp.json()
    return result["choices"][0]["message"]["content"]


async def generate_one_problem(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    problem: dict,
    k: int,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    prompt,
) -> tuple[list[str], list[str], list[list[dict]]]:
    """Generate k inputs sequentially; each iteration sees the inputs produced
    by the previous iterations of this call as priors.

    Returns: (inputs, raw_responses, iter_prompts).
    """
    prior_inputs: list[str] = []
    raw_responses: list[str] = []
    iter_prompts: list[list[dict]] = []
    for i in range(k):
        messages = build_messages(prompt, problem, prior_inputs, i, k)
        iter_prompts.append(messages)
        resp = await call_model(session, url, model_name, messages, max_tokens, temperature, enable_thinking)
        inp = extract_final_input(resp)
        prior_inputs.append(inp)
        raw_responses.append(resp)
    return prior_inputs, raw_responses, iter_prompts


def synthetic_combined_response(raw_responses: list[str], inputs: list[str], k: int) -> str:
    """Combine K per-iteration responses into a single markdown blob ending in a
    Final Summary JSON block."""
    parts: list[str] = []
    for i, raw in enumerate(raw_responses):
        parts.append(f"\n# === Iteration {i + 1} of {k} ===\n")
        parts.append(raw)
    summary_obj = {f"Input {i + 1}": x for i, x in enumerate(inputs)}
    parts.append(f"\n\n## Final Summary\n```json\n{json.dumps(summary_obj, indent=2, ensure_ascii=False)}\n```\n")
    return "\n".join(parts)


def _count_valid_unique(inputs: list[str]) -> int:
    """Count non-empty inputs after strip-equal dedup. Matches the dedup logic
    in test_rl_reward_manager.py's probe loader, so this counts exactly the
    K_eff the reward path will see."""
    seen: set[str] = set()
    count = 0
    for s in inputs:
        s_stripped = (s or "").strip()
        if not s_stripped or s_stripped in seen:
            continue
        seen.add(s_stripped)
        count += 1
    return count


async def main_async(args: argparse.Namespace) -> None:
    prompt = get_prompt(args.prompt_version)
    print(f"Using prompt version: {args.prompt_version}  (system head: {prompt.SYSTEM[:80]!r}...)")
    bm_dir = Path(args.benchmark_json_dir)
    assert bm_dir.is_dir(), f"benchmark_json_dir not found: {bm_dir}"
    paths = sorted(bm_dir.glob("*.json"))
    if args.limit is not None:
        paths = paths[: args.limit]
    problems: list[dict] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            problems.append(json.load(f))
    print(f"Loaded {len(problems)} problems")

    # Incremental mode: read existing parquet, skip qids that already have
    # >= args.k unique valid inputs (after strip+dedup matching the reward
    # loader), and for partial qids only generate the missing iters and merge.
    existing_data: dict[str, dict] = {}
    if args.incremental_from is not None:
        assert args.incremental_from.is_file(), (
            f"--incremental-from not a file: {args.incremental_from}"
        )
        df_existing = pd.read_parquet(args.incremental_from)
        required_cols = {"question_id", "iter_inputs", "iter_responses", "iter_prompts"}
        missing = required_cols - set(df_existing.columns)
        assert not missing, (
            f"--incremental-from parquet missing columns: {sorted(missing)}; "
            f"got {sorted(df_existing.columns)}"
        )
        has_prompt_version_col = "prompt_version" in df_existing.columns
        for _, row in df_existing.iterrows():
            qid = str(row["question_id"])
            def _tolist(x):
                return list(x.tolist()) if hasattr(x, "tolist") else list(x)
            existing_data[qid] = {
                "iter_inputs": _tolist(row["iter_inputs"]),
                "iter_responses": _tolist(row["iter_responses"]),
                "iter_prompts": _tolist(row["iter_prompts"]),
                # Parquets without a prompt_version column get an explicit
                # "unknown" tag rather than silently inheriting --prompt-version.
                "prompt_version": (
                    str(row["prompt_version"]) if has_prompt_version_col else "unknown"
                ),
            }
        print(f"[incremental] loaded existing data for {len(existing_data)} qids from {args.incremental_from}")

    per_prob_k_needed: dict[str, int] = {}
    n_skip = n_partial = n_full = 0
    for prob in problems:
        qid = str(prob["question_id"])
        ex = existing_data.get(qid)
        if ex is None:
            per_prob_k_needed[qid] = args.k
            n_full += 1
        else:
            existing_valid = _count_valid_unique(ex["iter_inputs"])
            k_needed = max(0, args.k - existing_valid)
            per_prob_k_needed[qid] = k_needed
            if k_needed == 0:
                n_skip += 1
            else:
                n_partial += 1
    if args.incremental_from is not None:
        print(f"[incremental] skip={n_skip} (have>={args.k}), partial-fill={n_partial}, full-gen={n_full}")

    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def gen_with_sem(prob: dict) -> tuple[dict, list[str] | None, list[str], list[list[dict]]]:
        nonlocal done
        qid = str(prob["question_id"])
        k_needed = per_prob_k_needed[qid]
        ex = existing_data.get(qid)

        if k_needed == 0:
            # Skip generation: pass through existing data unchanged.
            done += 1
            existing_valid = _count_valid_unique(ex["iter_inputs"])
            print(f"  [{done}/{len(problems)}] qid={qid} SKIP (existing K_eff={existing_valid})")
            return prob, ex["iter_inputs"], ex["iter_responses"], ex["iter_prompts"]

        async with sem:
            try:
                new_inputs, new_raws, new_prompts = await generate_one_problem(
                    session, url, args.model_name, prob, k_needed, args.max_tokens, args.temperature,
                    args.enable_thinking, prompt,
                )
            except Exception as e:
                done += 1
                print(f"  [{done}/{len(problems)}] qid={qid} FAILED: {type(e).__name__}: {e}")
                if ex is not None:
                    return prob, ex["iter_inputs"], ex["iter_responses"], ex["iter_prompts"]
                return prob, None, ["" for _ in range(args.k)], [[] for _ in range(args.k)]
        done += 1

        if ex is not None:
            merged_inputs = list(ex["iter_inputs"]) + list(new_inputs)
            merged_raws = list(ex["iter_responses"]) + list(new_raws)
            merged_prompts = list(ex["iter_prompts"]) + list(new_prompts)
            n_ok_new = sum(1 for x in new_inputs if x and x.strip())
            existing_valid = _count_valid_unique(ex["iter_inputs"])
            merged_valid = _count_valid_unique(merged_inputs)
            print(
                f"  [{done}/{len(problems)}] qid={qid}  +{k_needed} iters "
                f"(parsed_new={n_ok_new}/{k_needed})  K_eff {existing_valid}→{merged_valid}"
            )
            return prob, merged_inputs, merged_raws, merged_prompts

        n_ok = sum(1 for x in new_inputs if x)
        n_unique = len(set(x for x in new_inputs if x))
        print(f"  [{done}/{len(problems)}] qid={qid}  parsed={n_ok}/{k_needed}  unique={n_unique}/{k_needed}")
        return prob, new_inputs, new_raws, new_prompts

    async with aiohttp.ClientSession() as session:
        tasks = [gen_with_sem(p) for p in problems]
        results = await asyncio.gather(*tasks)

    # Fail-fast on excessive generation failures (>10% of the generated
    # problems raised or produced 0 parseable inputs). Skipped qids
    # (incremental, already have K_eff>=k) are excluded from the denominator.
    fillable = [
        (prob, inputs, raws, iter_prompts)
        for prob, inputs, raws, iter_prompts in results
        if per_prob_k_needed[str(prob["question_id"])] > 0
    ]
    n_total = len(fillable)
    n_exc = sum(1 for _, inputs, _, _ in fillable if inputs is None)
    n_parsed_zero = sum(
        1 for _, inputs, _, _ in fillable
        if inputs is not None and _count_valid_unique(inputs) == 0
    )
    n_bad = n_exc + n_parsed_zero
    fail_threshold = 0.10
    print(
        f"\nGeneration summary: {n_total} problems generated, "
        f"{n_exc} exception, {n_parsed_zero} parsed=0, "
        f"{n_total - n_bad} usable  ({len(results) - n_total} skipped via --incremental-from)"
    )
    if n_total > 0:
        assert n_bad <= int(fail_threshold * n_total), (
            f"Aug-input generation failed for {n_bad}/{n_total} generated problems "
            f"({n_bad / max(1, n_total):.1%}), exceeding the {fail_threshold:.0%} "
            f"threshold ({n_exc} exception + {n_parsed_zero} parsed=0). "
            "Refusing to write a degraded parquet. Investigate the server/prompt and rerun."
        )

    rows: list[dict] = []
    for prob, inputs, raws, iter_prompts in results:
        if inputs is None:
            # Exception case: still write a row (with empty inputs) so the qid
            # is present in the parquet.
            inputs = []
        k_actual = len(inputs)
        combined = synthetic_combined_response(raws, inputs, k_actual)
        # Per-row prompt_version lineage: pass-through rows keep their source
        # prompt_version; partial fills with a different prompt get "<old>+<current>".
        qid = str(prob["question_id"])
        k_needed = per_prob_k_needed.get(qid, args.k)
        ex = existing_data.get(qid)
        if k_needed == 0 and ex is not None:
            row_prompt_version = ex["prompt_version"]
        elif k_needed > 0 and ex is not None and ex["prompt_version"] != args.prompt_version:
            row_prompt_version = f"{ex['prompt_version']}+{args.prompt_version}"
        else:
            row_prompt_version = args.prompt_version
        rows.append({
            "data_source": "livecodebench_v6_aug_input_iter",
            "question_id": prob["question_id"],
            "question_title": prob.get("question_title", ""),
            "platform": prob.get("platform", ""),
            "difficulty": prob.get("difficulty", ""),
            "contest_date": prob.get("contest_date", ""),
            "prompt": [
                {"role": "system", "content": prompt.SYSTEM},
                {"role": "user", "content": f"(iterative generation, K={k_actual}; per-iteration prompts in iter_prompts column)"},
            ],
            "responses": [combined],
            "iter_prompts": iter_prompts,
            "iter_responses": raws,
            "iter_inputs": inputs,
            "prompt_version": row_prompt_version,
            "reward_model": {"style": "rule", "ground_truth": ""},
        })
    df = pd.DataFrame(rows)
    out: Path = args.output_parquet
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-json-dir", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32000,
                        help="Per-iteration generation cap (tokens).")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Request thinking-mode generation (chat_template_kwargs.enable_thinking=True).")
    parser.add_argument("--incremental-from", type=Path, default=None,
                        help="If set, read this existing parquet and only re-generate qids whose "
                             "unique-non-empty input count (post strip+dedup) is below --k. For partial "
                             "qids, generate (k - existing_K_eff) more iters and merge with existing "
                             "iter_inputs / iter_responses / iter_prompts. Qids with existing K_eff >= k "
                             "are passed through unchanged.")
    parser.add_argument("--prompt-version", default="v1",
                        help="Prompt version from recipe.code_bench.aug_input_prompts (default v1).")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
