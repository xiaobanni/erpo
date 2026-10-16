from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from recipe.code_bench.fmv_pointwise import capture_outputs_batch, compute_fmv_pointwise_scores
from recipe.code_bench.test_rl_reward_manager import CodeBenchTestRLRewardManager
from verl import DataProto
from verl.experimental.reward_loop.errors import RewardFatalError
from verl.trainer.ppo.ray_trainer import (
    _build_group_preserving_reward_chunks,
    _requires_trainer_batch_reward,
)


def _cfg(reward_kwargs=None, adv_estimator="grpo"):
    return OmegaConf.create({
        "reward": {
            "reward_model": {"enable": False, "enable_resource_pool": False},
            "reward_kwargs": reward_kwargs or {},
        },
        "algorithm": {"adv_estimator": adv_estimator},
    })


def test_requires_trainer_batch_reward_for_fmv_modes_and_side_channels():
    assert _requires_trainer_batch_reward(_cfg({"reward_mode": "fmv_pointwise"}))
    assert _requires_trainer_batch_reward(_cfg({"reward_mode": "fmv"}))
    assert _requires_trainer_batch_reward(_cfg({"reward_mode": "fmv-pointwise"}))
    assert _requires_trainer_batch_reward(
        _cfg({"reward_mode": "public_binary", "fmv_aug_inputs_parquet": "inputs.parquet"})
    )
    assert _requires_trainer_batch_reward(_cfg({"reward_mode": "public_binary"}, "grpo_fmv_penalty_only"))
    assert _requires_trainer_batch_reward(_cfg({"reward_mode": "public_binary"}, "nsr_pcr"))


def test_requires_trainer_batch_reward_false_for_pointwise_public_reward():
    assert not _requires_trainer_batch_reward(_cfg({"reward_mode": "public_binary"}, "grpo"))
    assert not _requires_trainer_batch_reward(_cfg({"reward_mode": "private_binary"}, "grpo"))


def test_group_preserving_reward_chunks_do_not_split_uid_groups():
    data = DataProto.from_dict(
        tensors={"x": torch.arange(6).view(6, 1)},
        non_tensors={"uid": np.array(["a", "a", "b", "c", "c", "c"], dtype=object)},
    )

    chunks = _build_group_preserving_reward_chunks(data, num_workers=2)
    chunk_indices = [indices for _chunk, indices in chunks]
    assert sorted(idx for indices in chunk_indices for idx in indices) == list(range(6))

    owner_by_uid = {}
    uid_values = data.non_tensor_batch["uid"].tolist()
    for chunk_id, indices in enumerate(chunk_indices):
        for idx in indices:
            uid = uid_values[idx]
            owner_by_uid.setdefault(uid, chunk_id)
            assert owner_by_uid[uid] == chunk_id


def test_capture_outputs_batch_pair_granularity_matches_candidate_granularity():
    codes = [
        "print(input().strip())",
        "print('constant')",
    ]
    inputs = ["alpha\n", "beta\n"]

    by_candidate = capture_outputs_batch(codes, inputs, None, timeout=2, max_workers=2, granularity="candidate")
    by_pair = capture_outputs_batch(codes, inputs, None, timeout=2, max_workers=2, granularity="pair")

    assert by_pair == by_candidate
    assert [[item["status"] for item in row] for row in by_pair] == [["ok", "ok"], ["ok", "ok"]]
    assert [[item["value"] for item in row] for row in by_pair] == [["alpha", "beta"], ["constant", "constant"]]


def test_compute_fmv_pointwise_scores_uses_group_consensus_not_single_candidate_mode():
    inputs = ["x1", "x2", "x2", ""]
    outputs = []
    for idx in range(16):
        first = {"status": "ok", "value": repr("A" if idx < 9 else "B")}
        second = {"status": "ok", "value": repr("C" if idx % 2 == 0 else "D")}
        if idx == 13:
            first = {"status": "timeout", "value": ""}
        outputs.append([first, second, {"status": "ok", "value": "ignored"}, {"status": "ok", "value": "ignored"}])

    scores, k_eff, diag = compute_fmv_pointwise_scores(inputs, outputs)

    assert k_eff == 2
    assert diag["n_dedup_dropped"] == 2
    assert scores[0] == 1.0  # A is majority on x1; C/D are tied modes on x2.
    assert scores[10] == 0.5  # B loses x1 but D is tied on x2.
    assert scores[13] == 0.5  # timeout is excluded from x1 mode and earns no x1 credit.


def test_compute_fmv_for_group_scores_multiple_candidates():
    manager = CodeBenchTestRLRewardManager.__new__(CodeBenchTestRLRewardManager)
    manager.fmv_use_private_inputs = False
    manager.fmv_aug_inputs = {"dummy": ["b\n"]}
    manager.fmv_timeout = 2
    manager.fmv_max_workers = 1
    manager.fmv_exec_granularity = "pair"

    benchmark = SimpleNamespace(
        public_test_cases=[SimpleNamespace(input="a\n")],
        private_test_cases=[],
        metadata={},
    )
    response_texts = [
        "```python\nprint(input().strip())\n```",
        "```python\nprint(input().strip())\n```",
        "```python\nprint('constant')\n```",
    ]

    result = asyncio.run(
        manager._compute_fmv_for_group(
            question_id="dummy",
            benchmark=benchmark,
            response_texts=response_texts,
            n_rows=3,
            validate=False,
        )
    )

    assert result["K_eff"] == 2
    assert result["scores"] == [1.0, 1.0, 0.0]


def test_fmv_training_rejects_single_candidate_group():
    manager = CodeBenchTestRLRewardManager.__new__(CodeBenchTestRLRewardManager)

    with pytest.raises(RewardFatalError, match="group-aware training requires multiple same-uid candidates"):
        asyncio.run(
            manager._compute_fmv_for_group(
                question_id="dummy",
                benchmark=None,
                response_texts=["```python\nprint(1)\n```"],
                n_rows=1,
                validate=False,
            )
        )
