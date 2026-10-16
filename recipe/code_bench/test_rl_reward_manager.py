"""
Group-aware reward manager for code test-time RL.

Samples that share a `uid` (the rollouts of one prompt) are scored together.
Supported `reward_mode` values:

- `public_binary`: reward is 1.0 iff the candidate passes all public test cases.
- `fmv_pointwise`: PCR reward in [0, 1]. Every candidate of the group is run on
  a pooled input set (public test inputs + LLM-generated probe inputs); the
  reward is the fraction of inputs on which the candidate's output agrees with
  the group's majority output (see `recipe.code_bench.fmv_pointwise`).

During validation the private (hidden) tests are also executed so that
`full_all_passed` can drive the pass@k validation metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from recipe.code_bench.livecodebench import build_private_in_outs, build_public_in_outs, extract_code_from_response
from recipe.code_bench.reward_score import (
    REWARD_MODE_FMV_POINTWISE,
    REWARD_MODE_PUBLIC_BINARY,
    _extract_question_id,
    _normalize_reward_mode,
    evaluate_code_against_in_outs,
    load_benchmark,
    summarize_test_results,
)
from verl import DataProto
from verl.experimental.reward_loop.errors import RewardFatalError
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SUPPORTED_REWARD_MODES = (REWARD_MODE_PUBLIC_BINARY, REWARD_MODE_FMV_POINTWISE)


def _resolve_reward_kwargs(raw_value: Any) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, DictConfig):
        resolved = OmegaConf.to_container(raw_value, resolve=True)
    else:
        resolved = raw_value
    if resolved is None:
        return {}
    if not isinstance(resolved, dict):
        raise TypeError(f"reward.reward_kwargs must resolve to a dict, got {type(resolved)}")
    return dict(resolved)


class CodeBenchTestRLRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        reward_kwargs = _resolve_reward_kwargs(config.reward.get("reward_kwargs"))

        self.reward_mode = _normalize_reward_mode(reward_kwargs.get("reward_mode", REWARD_MODE_PUBLIC_BINARY))
        if self.reward_mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(
                f"reward.reward_kwargs.reward_mode must be one of {SUPPORTED_REWARD_MODES}, got {self.reward_mode!r}"
            )
        self.benchmark_dir = reward_kwargs.get("benchmark_dir")
        if not isinstance(self.benchmark_dir, str) or not self.benchmark_dir:
            raise ValueError("reward.reward_kwargs.benchmark_dir is required")

        self.timeout = int(reward_kwargs.get("timeout", 6))
        ec_raw = reward_kwargs.get("evaluation_concurrency", 1)
        if str(ec_raw).strip().lower() == "auto":
            num_workers = getattr(config.reward, "num_workers", 1)
            cpu_count = os.cpu_count() or 1
            self.evaluation_concurrency = max(1, cpu_count // num_workers)
        else:
            self.evaluation_concurrency = max(1, int(ec_raw))
        self._evaluation_semaphore = asyncio.Semaphore(self.evaluation_concurrency)

        # Optional per-sample JSONL dump (one file per step and phase).
        save_data_dir_raw = reward_kwargs.get("save_data_dir")
        self.save_data_dir: Path | None = None
        if save_data_dir_raw:
            self.save_data_dir = Path(str(save_data_dir_raw))
            self.save_data_dir.mkdir(parents=True, exist_ok=True)
        self._save_data_lock = threading.Lock()

        # PCR (fmv_pointwise): load the pre-generated probe inputs (parquet from
        # generate_aug_inputs_iterative.py). One row per question_id with an
        # `iter_inputs` column (list of raw input strings).
        self.fmv_aug_inputs: dict[str, list[str]] = {}
        self.fmv_timeout = int(reward_kwargs.get("fmv_timeout", 5))
        self.fmv_max_workers = int(reward_kwargs.get("fmv_max_workers", 16))
        self.fmv_exec_granularity = str(reward_kwargs.get("fmv_exec_granularity", "pair")).strip().lower()
        allowed_fmv_exec_granularities = {
            "candidate",
            "candidate_level",
            "pair",
            "pairwise",
            "candidate_input",
            "input",
        }
        if self.fmv_exec_granularity not in allowed_fmv_exec_granularities:
            raise ValueError(
                f"reward.reward_kwargs.fmv_exec_granularity must be one of "
                f"{sorted(allowed_fmv_exec_granularities)}, got {self.fmv_exec_granularity!r}"
            )
        if self.reward_mode == REWARD_MODE_FMV_POINTWISE:
            import pandas as pd

            fmv_path_raw = reward_kwargs.get("fmv_aug_inputs_parquet")
            if not isinstance(fmv_path_raw, str) or not fmv_path_raw:
                raise ValueError(
                    "reward.reward_kwargs.fmv_aug_inputs_parquet is required for reward_mode=fmv_pointwise "
                    "(path to parquet produced by recipe.code_bench.generate_aug_inputs_iterative)"
                )
            fmv_path = Path(fmv_path_raw)
            if not fmv_path.is_file():
                raise ValueError(f"fmv_aug_inputs_parquet not found: {fmv_path}")
            df = pd.read_parquet(fmv_path)
            if "question_id" not in df.columns or "iter_inputs" not in df.columns:
                raise ValueError(
                    f"fmv_aug_inputs_parquet missing required columns; got {list(df.columns)}; "
                    "expected at least question_id + iter_inputs"
                )
            # Fail-fast on duplicate question_id rows: silent last-writer-wins
            # would mask a data pipeline bug.
            df_qid = df["question_id"].astype(str)
            dup_qids = df_qid[df_qid.duplicated()].unique().tolist()
            if dup_qids:
                preview = ", ".join(dup_qids[:5])
                more = "" if len(dup_qids) <= 5 else f" (+{len(dup_qids) - 5} more)"
                raise ValueError(
                    f"fmv_aug_inputs_parquet has duplicate question_id rows: "
                    f"{preview}{more}. Regenerate {fmv_path} with unique qids."
                )
            for _, row in df.iterrows():
                qid = str(row["question_id"])
                inps = row["iter_inputs"]
                if hasattr(inps, "tolist"):
                    inps = inps.tolist()
                # Drop empties + dedup by strip-equal (matches fmv_pointwise.dedupe_inputs)
                seen: set[str] = set()
                kept_inputs: list[str] = []
                for s in inps:
                    s_stripped = (s or "").strip()
                    if not s_stripped or s_stripped in seen:
                        continue
                    seen.add(s_stripped)
                    kept_inputs.append(s)
                self.fmv_aug_inputs[qid] = kept_inputs
            # Empty probes for a qid fall back to a public-only voting pool
            # (weaker signal but non-zero). A large fraction of empty qids
            # almost certainly means the wrong parquet was loaded.
            empty_qids = [qid for qid, k in self.fmv_aug_inputs.items() if not k]
            if empty_qids:
                total = len(self.fmv_aug_inputs)
                empty_frac = len(empty_qids) / total
                preview = ", ".join(empty_qids[:5])
                more = "" if len(empty_qids) <= 5 else f" (+{len(empty_qids) - 5} more)"
                if empty_frac > 0.1:
                    raise ValueError(
                        f"[FMV] {len(empty_qids)}/{total} qids ({empty_frac:.1%}) have 0 "
                        f"kept aug inputs after dedup: {preview}{more}. "
                        f"Exceeds 10% tolerance — likely wrong parquet file or loader bug. "
                        f"Regenerate {fmv_path} or fix the generation pipeline."
                    )
                print(
                    f"[FMV] WARNING: {len(empty_qids)}/{total} qids ({empty_frac:.1%}) have "
                    f"0 kept aug inputs: {preview}{more}. Within 10% tolerance — these qids "
                    f"will use public-only FMV voting pool (signal weaker but non-zero)."
                )
            print(f"[FMV] loaded aug inputs for {len(self.fmv_aug_inputs)} qids from {fmv_path}")

    async def run_single(self, data: DataProto) -> dict:
        outputs = await self.run_batch(data)
        return outputs[0]

    async def run_batch(self, data: DataProto) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any] | None] = [None for _ in range(len(data))]
        for row_indices in self._group_row_indices(data):
            group_outputs = await self._score_group(data, row_indices)
            for row_idx, output in zip(row_indices, group_outputs, strict=True):
                outputs[row_idx] = output

        if any(output is None for output in outputs):
            raise RewardFatalError("Some reward outputs are missing after group scoring")
        return [output for output in outputs if output is not None]

    def _group_row_indices(self, data: DataProto) -> list[list[int]]:
        uid_values = data.non_tensor_batch.get("uid")
        sample_order = data.non_tensor_batch.get("rollout_sample_idx")

        grouped: dict[object, list[int]] = {}
        ordered_uids: list[object] = []
        for row_idx in range(len(data)):
            uid = row_idx if uid_values is None else uid_values[row_idx]
            if uid not in grouped:
                grouped[uid] = []
                ordered_uids.append(uid)
            grouped[uid].append(row_idx)

        def sort_key(row_idx: int) -> tuple[int, int]:
            if sample_order is None:
                return (row_idx, row_idx)
            return (int(sample_order[row_idx]), row_idx)

        return [sorted(grouped[uid], key=sort_key) for uid in ordered_uids]

    def _decode_response(self, data_item) -> str:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
        if valid_response_length <= 0:
            return ""
        valid_response_ids = response_ids[:valid_response_length]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    def _decode_prompt(self, data_item) -> str:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        attention_mask = data_item.batch["attention_mask"][:prompt_length]
        valid_prompt_length = int(attention_mask.sum().item())
        if valid_prompt_length <= 0:
            return ""
        # Prompts are left-padded; valid tokens are at the right.
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        return self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

    def _question_id_for_group(self, data: DataProto, row_indices: list[int]) -> str:
        question_ids = {
            _extract_question_id(data[row_idx].non_tensor_batch["reward_model"]["ground_truth"]) for row_idx in row_indices
        }
        if len(question_ids) != 1:
            raise RewardFatalError(
                f"Expected a single question_id within a uid group, got {sorted(question_ids)}"
            )
        return next(iter(question_ids))

    async def _evaluate_response(
        self,
        solution_str: str,
        in_outs: dict[str, Any],
        *,
        complete_evaluation: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        """Run the solution against a set of test cases.

        Set `complete_evaluation=False` to stop at the first failing test — this is
        correct whenever the caller only needs the overall all-passed boolean
        (e.g. private tests feeding `full_all_passed`).
        """
        async with self._evaluation_semaphore:
            return await self.loop.run_in_executor(
                None,
                lambda: evaluate_code_against_in_outs(
                    solution_str,
                    in_outs,
                    timeout=self.timeout,
                    complete_evaluation=complete_evaluation,
                ),
            )

    async def _score_group(self, data: DataProto, row_indices: list[int]) -> list[dict[str, Any]]:
        question_id = self._question_id_for_group(data, row_indices)
        validate = bool(data.meta_info.get("validate", False))

        benchmark = load_benchmark(question_id, self.benchmark_dir)
        public_in_outs = build_public_in_outs(benchmark)
        response_texts = [self._decode_response(data[row_idx]) for row_idx in row_indices]
        assert "global_steps" in data.meta_info, (
            "meta_info is missing 'global_steps'. Ensure the caller (e.g. ray_trainer._validate) "
            "sets meta_info['global_steps'] before dispatching to the reward manager."
        )
        global_steps = int(data.meta_info["global_steps"])

        public_eval_results = await asyncio.gather(
            *(self._evaluate_response(response_text, public_in_outs) for response_text in response_texts)
        )
        public_successes = [result[0] for result in public_eval_results]
        public_summaries = [summarize_test_results(result[1]["results"]) for result in public_eval_results]

        full_all_passed: list[bool] | None = None
        # Validation also runs the private tests (for the pass@k metrics).
        # Public tests are a subset of the full test set, so a public-failing
        # sample cannot pass the full set and its private run is skipped.
        if validate:
            full_all_passed = [False for _ in row_indices]
            if len(benchmark.private_test_cases) == 0:
                full_all_passed = [bool(public_success) for public_success in public_successes]
            else:
                private_in_outs = build_private_in_outs(benchmark)
                private_eval_indices: list[int] = []
                private_eval_tasks = []
                for local_idx, (public_success, response_text) in enumerate(
                    zip(public_successes, response_texts, strict=True)
                ):
                    if not public_success:
                        continue
                    private_eval_indices.append(local_idx)
                    private_eval_tasks.append(
                        self._evaluate_response(
                            response_text, private_in_outs, complete_evaluation=False
                        )
                    )

                private_eval_results = await asyncio.gather(*private_eval_tasks)
                for local_idx, (private_success, _private_metadata) in zip(
                    private_eval_indices,
                    private_eval_results,
                    strict=True,
                ):
                    full_all_passed[local_idx] = bool(private_success)

        if self.reward_mode == REWARD_MODE_FMV_POINTWISE:
            # PCR: reward from behavioral consensus on the pooled (public + probe)
            # inputs. Public pass/fail and private flags are only reported via
            # reward_extra_info; they are not part of the reward.
            fmv_out = await self._compute_fmv_for_group(
                question_id=question_id,
                benchmark=benchmark,
                response_texts=response_texts,
                n_rows=len(row_indices),
                validate=validate,
            )
            scores = fmv_out["scores"]
            outputs = []
            for local_idx, (score, public_success, public_summary) in enumerate(
                zip(scores, public_successes, public_summaries, strict=True)
            ):
                out = self._build_reward_output(
                    score=float(score),
                    question_id=question_id,
                    public_success=public_success,
                    public_summary=public_summary,
                    full_all_passed=None if full_all_passed is None else full_all_passed[local_idx],
                )
                out["reward_extra_info"]["fmv_K_eff"] = float(fmv_out["K_eff"])
                out["reward_extra_info"]["fmv_score"] = float(score)
                out["reward_extra_info"]["fmv_n_aug"] = float(fmv_out["n_aug"])
                out["reward_extra_info"]["fmv_n_public"] = float(fmv_out["n_public"])
                out["reward_extra_info"]["fmv_n_combined"] = float(fmv_out["n_combined"])
                outputs.append(out)
        else:
            outputs = [
                self._build_reward_output(
                    score=float(public_success),
                    question_id=question_id,
                    public_success=public_success,
                    public_summary=public_summary,
                    full_all_passed=None if full_all_passed is None else full_all_passed[local_idx],
                )
                for local_idx, (public_success, public_summary) in enumerate(
                    zip(public_successes, public_summaries, strict=True)
                )
            ]

        self._maybe_dump_samples(
            data=data,
            row_indices=row_indices,
            outputs=outputs,
            question_id=question_id,
            response_texts=response_texts,
            public_in_outs=public_in_outs,
            full_all_passed=full_all_passed,
            validate=validate,
            global_steps=global_steps,
        )
        return outputs

    def _build_reward_output(
        self,
        *,
        score: float,
        question_id: str,
        public_success: bool,
        public_summary: dict[str, Any],
        full_all_passed: bool | None = None,
    ) -> dict[str, Any]:
        reward_extra_info = {
            "score": float(score),
            "acc": float(score),
            "reward_mode": self.reward_mode,
            "public_all_passed": float(public_success),
            "public_pass_rate": float(public_summary["pass_rate"]),
            "num_public_tests": float(public_summary["num_tests"]),
            "num_public_tests_passed": float(public_summary["num_passed"]),
        }
        if full_all_passed is not None:
            reward_extra_info["full_all_passed"] = float(full_all_passed)
        return {"reward_score": float(score), "reward_extra_info": reward_extra_info}

    async def _compute_fmv_for_group(
        self,
        *,
        question_id: str,
        benchmark,
        response_texts: list[str],
        n_rows: int,
        validate: bool = False,
    ) -> dict[str, Any]:
        """Compute PCR (FMV-pointwise) scores for one GRPO group: consensus on
        pooled public + probe inputs.
        Returns {scores: list[float], K_eff, n_aug, n_public, n_combined}.
        """
        if len(response_texts) != n_rows:
            raise RewardFatalError(
                f"[FMV] response_texts length mismatch for question_id={question_id!r}: "
                f"len(response_texts)={len(response_texts)}, n_rows={n_rows}"
            )
        if not validate and n_rows <= 1:
            raise RewardFatalError(
                f"[FMV] group-aware training requires multiple same-uid candidates; "
                f"got n_rows={n_rows} for question_id={question_id!r}. "
                "Disable agent-loop per-sample reward and compute FMV after full batch rollout."
            )

        from recipe.code_bench.fmv_pointwise import (
            capture_outputs_batch,
            compute_fmv_pointwise_scores,
        )

        # Fail-fast if this qid is missing from the probe parquet: a silent
        # empty fallback would mix probe-aware and probe-blind groups.
        if question_id not in self.fmv_aug_inputs:
            raise RewardFatalError(
                f"[FMV] question_id={question_id!r} not present in "
                f"fmv_aug_inputs_parquet. The probe parquet must cover the "
                f"full training qid set; regenerate it over the correct benchmark split."
            )
        aug_inputs = self.fmv_aug_inputs[question_id]
        public_inputs = [t.input for t in benchmark.public_test_cases]
        # Dedup combined list by strip-equal (matches loader dedup). Keep public
        # inputs first so their order is stable across runs.
        seen: set[str] = set()
        combined_inputs: list[str] = []
        for s in [*public_inputs, *aug_inputs]:
            s_stripped = (s or "").strip()
            if not s_stripped or s_stripped in seen:
                continue
            seen.add(s_stripped)
            combined_inputs.append(s)

        fn_name = (
            benchmark.metadata.get("func_name")
            if benchmark.metadata else None
        )
        # combined_inputs is essentially always non-empty. The guard catches a
        # problem with 0 public tests AND all probe inputs dropped by dedup.
        if not combined_inputs:
            logger.warning(
                "[FMV] qid=%s has 0 combined inputs (aug=%d, public=%d); "
                "returning 0 score for all %d candidates in this group.",
                question_id, len(aug_inputs), len(public_inputs), n_rows,
            )
            return {
                "scores": [0.0] * n_rows,
                "K_eff": 0,
                "n_aug": len(aug_inputs),
                "n_public": len(public_inputs),
                "n_combined": 0,
            }
        codes = [extract_code_from_response(rt) for rt in response_texts]
        loop = asyncio.get_running_loop()
        per_cand_outs = await loop.run_in_executor(
            None,
            lambda: capture_outputs_batch(
                codes, combined_inputs, fn_name,
                self.fmv_timeout, self.fmv_max_workers,
                granularity=self.fmv_exec_granularity,
            ),
        )
        scores, K_eff, _diag = compute_fmv_pointwise_scores(combined_inputs, per_cand_outs)
        return {
            "scores": scores,
            "K_eff": K_eff,
            "n_aug": len(aug_inputs),
            "n_public": len(public_inputs),
            "n_combined": len(combined_inputs),
        }

    def _maybe_dump_samples(
        self,
        *,
        data: DataProto,
        row_indices: list[int],
        outputs: list[dict[str, Any]],
        question_id: str,
        response_texts: list[str],
        public_in_outs: dict[str, Any],
        full_all_passed: list[bool] | None,
        validate: bool,
        global_steps: int,
    ) -> None:
        """Append per-sample records to a JSONL file under save_data_dir.

        Each record contains the prompt, response, reward, public test cases,
        and (if validation) the private test pass result. One file per
        (step, train|val) phase.
        """
        if self.save_data_dir is None:
            return

        public_inputs = public_in_outs.get("inputs", []) or []
        public_outputs = public_in_outs.get("outputs", []) or []
        fn_name = public_in_outs.get("fn_name")
        public_test_payload = [
            {"input": _stringify_io(test_in), "output": _stringify_io(test_out)}
            for test_in, test_out in zip(public_inputs, public_outputs, strict=False)
        ]

        rollout_sample_idx_arr = data.non_tensor_batch.get("rollout_sample_idx")

        records: list[dict[str, Any]] = []
        for local_idx, row_idx in enumerate(row_indices):
            data_item = data[row_idx]
            prompt_text = self._decode_prompt(data_item)
            rollout_idx: int | None = None
            if rollout_sample_idx_arr is not None:
                rollout_idx = int(rollout_sample_idx_arr[row_idx])

            record: dict[str, Any] = {
                "step": global_steps,
                "phase": "val" if validate else "train",
                "question_id": question_id,
                "rollout_sample_idx": rollout_idx,
                "row_idx": int(row_idx),
                "prompt": prompt_text,
                "response": response_texts[local_idx],
                "reward": float(outputs[local_idx]["reward_score"]),
                "reward_extra_info": outputs[local_idx]["reward_extra_info"],
                "public_test_cases": public_test_payload,
                "fn_name": fn_name,
            }
            if full_all_passed is not None:
                record["full_all_passed"] = bool(full_all_passed[local_idx])
            records.append(record)

        phase = "val" if validate else "train"
        assert global_steps >= 0, f"global_steps must be non-negative for jsonl dump, got {global_steps}"
        fname = self.save_data_dir / f"step_{global_steps:06d}_{phase}.jsonl"
        with self._save_data_lock:
            with open(fname, "a") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False, default=str))
                    f.write("\n")


def _stringify_io(value: Any) -> str:
    """Best-effort string serialization of a test input/output payload."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
