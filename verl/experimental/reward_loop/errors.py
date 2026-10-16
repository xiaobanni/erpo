class RewardFatalError(Exception):
    """Step-time fail-fast signal that must halt training.

    Reward-side wrappers (RewardLoopWorker.compute_score / compute_score_batch,
    AgentLoopWorker async-reward dispatch) catch broad ``Exception`` to keep
    transient failures (network blips, one-off subprocess races) from killing a
    long-running training run — they convert such errors into a zero-reward
    sentinel with ``reward_fallback=1.0`` logged for observability.

    That broad catch must NOT swallow config / data / routing bugs that the
    reward manager raises on purpose to force a loud crash, e.g.::

        - aug-input parquet missing the current question_id
        - GRPO group size collapsed to 1 (per-sample reward routed where
          group reward was expected)
        - response/group-size mismatch after batch rollout

    Raise ``RewardFatalError`` from such guards; wrapper layers re-raise this
    type unchanged.
    """
