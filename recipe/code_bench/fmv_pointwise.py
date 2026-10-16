"""PCR (FMV-pointwise) core algorithm: dedup + canonicalize + execute + score.

Used by recipe.code_bench.test_rl_reward_manager (reward_mode=fmv_pointwise).

Algorithm:
- Input dedup: strip-equal, drop empties → K_effective ≤ K_gen
- Output canonicalization: ast.literal_eval + float round + dict sort (fallback raw strip)
- Per-input mode (computed once per input)
- Errors / no-code EXCLUDED from mode voting, candidate gets 0 for that input
- Tied modes all credited
- score_i = matches / K_effective ∈ [0, 1]

Executor design:
- Each execution runs in a fresh forked subprocess, watched by a main-thread
  watchdog that SIGKILLs it on deadline (see `_capture_one_candidate` /
  `_capture_one_candidate_input`).
- The subprocess pre-imports the benchmark harness preamble and applies
  `reliability_guard` before running candidate code.
- SIGALRM-based per-input timeout inside the subprocess (Unix only).
"""

from __future__ import annotations

import ast
import multiprocessing as mp
import pickle
import signal
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


# Cap per-output value size before sending across mp.Pipe.
# `multiprocessing.Pipe` on Linux uses `os.pipe` with ~64 KB default capacity.
# A candidate that returns a huge object (e.g. `list(range(10**6))`) produces
# a multi-MB repr; pickling + sending it can fill the pipe buffer and block
# the child in `conn.send()`. The parent only drains the pipe AFTER the child
# exits, so the child blocks forever, gets watchdog-SIGKILLed, and the K
# outputs are lost as `subprocess_exited_exitcode=-9`. Truncating here keeps
# values informative for FMV consensus (canonicalize_output handles strings
# robustly) while ensuring a full K-output payload fits comfortably under
# the pipe capacity even for K=10.
_MAX_VALUE_LEN = 4096
_PAIR_WATCHDOG_SLACK_SECONDS = 5


SENTINEL_NO_CODE = "__FMV_NO_CODE__"
SENTINEL_RUN_ERROR = "__FMV_RUN_ERROR__"
SENTINEL_PARSE_ERROR = "__FMV_PARSE_ERROR__"
SENTINEL_TIMEOUT = "__FMV_TIMEOUT__"
SENTINEL_NO_MARKER = "__FMV_NO_MARKER__"
SENTINEL_MISSING = "__FMV_MISSING__"

ALL_SENTINELS = {
    SENTINEL_NO_CODE, SENTINEL_RUN_ERROR, SENTINEL_PARSE_ERROR,
    SENTINEL_TIMEOUT, SENTINEL_NO_MARKER, SENTINEL_MISSING,
}


def canonicalize_output(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return ""
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return s
    return repr(_normalize_value(v))


def _normalize_value(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, dict):
        return {k: _normalize_value(val) for k, val in sorted(v.items(), key=lambda kv: repr(kv[0]))}
    if isinstance(v, (list, tuple)):
        normalized = [_normalize_value(x) for x in v]
        return type(v)(normalized)
    return v


def status_to_sentinel(status: str) -> str | None:
    mapping = {
        "no_code": SENTINEL_NO_CODE,
        "run_error": SENTINEL_RUN_ERROR,
        "parse_error": SENTINEL_PARSE_ERROR,
        "timeout": SENTINEL_TIMEOUT,
        "no_marker": SENTINEL_NO_MARKER,
    }
    return mapping.get(status)


def is_valid_output(canonical_or_sentinel: str) -> bool:
    return canonical_or_sentinel not in ALL_SENTINELS


def dedupe_inputs(inputs: list[str]) -> list[int]:
    """Return list of indices to KEEP after strip-equal dedup. Empty inputs dropped."""
    seen: set[str] = set()
    keep: list[int] = []
    for i, inp in enumerate(inputs):
        s = (inp or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        keep.append(i)
    return keep


def compute_fmv_pointwise_scores(
    aug_inputs: list[str],
    aug_outputs_per_candidate: list[list[dict]],
) -> tuple[list[float], int, dict]:
    """See module docstring."""
    N = len(aug_outputs_per_candidate)
    if N == 0:
        return [], 0, {"reason": "no_candidates"}

    keep_indices = dedupe_inputs(aug_inputs)
    K_eff = len(keep_indices)
    if K_eff == 0:
        return [0.0] * N, 0, {"reason": "no_valid_inputs"}

    canonical: list[list[str]] = []
    for i in range(N):
        per_input: list[str] = []
        cand_outs = aug_outputs_per_candidate[i]
        for k_orig in keep_indices:
            if k_orig >= len(cand_outs) or cand_outs[k_orig] is None:
                per_input.append(SENTINEL_MISSING)
                continue
            ao = cand_outs[k_orig]
            sentinel = status_to_sentinel(ao.get("status", "missing"))
            if sentinel is not None:
                per_input.append(sentinel)
            else:
                per_input.append(canonicalize_output(ao.get("value", "")))
        canonical.append(per_input)

    tied_modes_per_input: list[set[str]] = []
    diag_per_input: list[dict] = []
    for k_local in range(K_eff):
        valid_outs = [canonical[i][k_local] for i in range(N) if is_valid_output(canonical[i][k_local])]
        if not valid_outs:
            tied_modes_per_input.append(set())
            diag_per_input.append({"n_valid": 0, "mode_count": 0, "n_tied": 0})
            continue
        counts = Counter(valid_outs)
        mode_count = max(counts.values())
        tied = {o for o, c in counts.items() if c == mode_count}
        tied_modes_per_input.append(tied)
        diag_per_input.append({"n_valid": len(valid_outs), "mode_count": mode_count, "n_tied": len(tied)})

    scores: list[float] = []
    for i in range(N):
        hits = sum(1 for k_local in range(K_eff)
                   if is_valid_output(canonical[i][k_local])
                   and canonical[i][k_local] in tied_modes_per_input[k_local])
        scores.append(hits / K_eff)

    diagnostics = {
        "K_gen": len(aug_inputs),
        "K_eff": K_eff,
        "n_dedup_dropped": len(aug_inputs) - K_eff,
        "per_input": diag_per_input,
    }
    return scores, K_eff, diagnostics


# ---- Per-candidate spawn+watchdog process model (mirrors check_correctness) ----
#
# Why not a ProcessPoolExecutor: that pattern relies on
# in-worker SIGALRM to bound per-task wall time. That fails silently when a
# candidate's code masks signal delivery (most common: `@unittest.mock.patch`
# decorator chains, or tight C loops that don't yield to Python). When SIGALRM
# can't fire, the worker hangs at 100% CPU forever, and as_completed in main
# blocks indefinitely — there is no per-future SIGKILL API on
# ProcessPoolExecutor. Recovery required manual SIGKILL of the worker PID,
# which is fine for offline scoring but a deal-breaker for training.
#
# This file mirrors `recipe/code_bench/reward_score.py:check_correctness`'s
# proven pattern: spawn one fresh subprocess per candidate, run all K aug
# inputs sequentially inside it, and have a main-thread watchdog poll
# `process.is_alive()` + `process.kill()` (OS-level SIGKILL, cannot be
# bypassed by user code). The per-task SIGALRM inside the worker stays as a
# cheap inner defense (catches the 99% of normal timeouts fast); the
# subprocess-level watchdog is the outer defense for the SIGALRM-bypass case.

# Filled by _worker_init in each spawned subprocess. Used as a pre-warmed
# namespace for user code exec so common stdlib imports (math, heapq,
# collections, typing, itertools, ...) do not re-cost per task. NOT accessed
# from main process.
_PREAMBLE: dict[str, Any] = {}


def _worker_init() -> None:
    """Per-subprocess init: pre-populate _PREAMBLE + neutralize dangerous OS calls.

    Mirrors the benchmark harness preamble in `recipe/code_bench/testing_util.py`
    (the `import_string` constant) so candidates that rely on harness-injected
    bare names (e.g., `lru_cache`, bare `sys`, `from functools import *` symbols)
    resolve identically under FMV scoring and benchmark evaluation. Without this
    alignment, FMV would mark otherwise-correct candidates as `run_error` and
    silently diverge from benchmark verdicts.

    Called once at the start of each spawned candidate subprocess (NOT in main).
    """
    global _PREAMBLE
    # Lazy import to keep module-load cost low; this runs once per subprocess.
    from recipe.code_bench.testing_util import import_string as _testing_util_imports
    from recipe.code_bench.testing_util import reliability_guard
    ns: dict[str, Any] = {}
    exec(_testing_util_imports, ns)
    _PREAMBLE = ns
    # Neutralize dangerous os/subprocess/shutil names in this subprocess so
    # untrusted model-generated code cannot escape via os.kill, subprocess.Popen,
    # shutil.rmtree, os.unlink, etc. Same mitigation as testing_util.run_test
    # applies for benchmark execution. Subprocess scope means this does not
    # affect the parent Ray actor.
    # RLIMIT_AS=8GB caps per-subprocess virtual memory: pathological candidate
    # code on LLM-generated aug inputs (e.g. `[0] * int(input())` with weird input)
    # would otherwise alloc until host OOM. With 8GB cap, MemoryError → run_error,
    # bounded host impact = 8 reward_actors × 4 workers × 8GB = 256GB worst case.
    reliability_guard(maximum_memory_bytes=8 * 1024**3)


# ---- Arg parsing helpers (functional path, run in worker subprocess) ----

def _try_split_eval(s: str):
    parts = [p for p in s.split("\n") if p.strip()]
    if len(parts) > 1:
        return [ast.literal_eval(p.strip()) for p in parts]
    raise ValueError("only one part after newline split")


def _try_single_literal(s: str):
    return [ast.literal_eval(s.strip())]


def _try_space_bracket(s: str):
    tokens, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch.isspace() and depth == 0:
            if cur:
                tokens.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    return [ast.literal_eval(t) for t in tokens]


def _parse_args_robust(s: str):
    s = s.strip()
    s_norm = s.replace("\\\\n", "\n").replace("\\n", "\n")
    s_norm = s_norm.replace("\\\\t", "\t").replace("\\t", "\t")
    strategies = [
        lambda: _try_split_eval(s),
        lambda: _try_split_eval(s_norm),
        lambda: _try_single_literal(s),
        lambda: _try_single_literal(s_norm),
        lambda: _try_space_bracket(s),
        lambda: _try_space_bracket(s_norm),
    ]
    last_err: Exception | None = None
    for strat in strategies:
        try:
            return strat()
        except Exception as e:
            last_err = e
    raise last_err if last_err else ValueError("no parse strategy succeeded")


# ---- In-worker execution ----

def _worker_run_one(code: str, aug_input: str, fn_name: str | None, timeout: int) -> dict:
    """Execute one (code, aug_input) in this worker process. Returns {status, value}.

    Dispatches by fn_name: non-empty → functional (Solution().fn_name(*args)),
    empty/None → STDIN (aug_input piped to sys.stdin, sys.stdout captured).

    SIGALRM-based timeout (Unix only — workers are subprocess main threads).
    """
    if not (code or "").strip():
        return {"status": "no_code", "value": ""}

    def _timeout_handler(signum, frame):
        raise TimeoutError("FMV exec timeout")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(max(1, int(timeout)))
    try:
        if fn_name:
            return _run_functional(code, aug_input, fn_name)
        return _run_stdin(code, aug_input)
    except TimeoutError:
        return {"status": "timeout", "value": ""}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _run_functional(code: str, aug_input: str, fn_name: str) -> dict:
    try:
        args = _parse_args_robust(aug_input)
    except TimeoutError:
        raise
    except Exception as e:
        return {"status": "parse_error", "value": f"{type(e).__name__}: {e}"[:300]}

    ns = dict(_PREAMBLE)
    ns["__name__"] = "__main__"
    try:
        exec(code, ns)
    except TimeoutError:
        raise
    except Exception as e:
        return {"status": "run_error", "value": f"{type(e).__name__}: {e}"[:300]}

    Solution = ns.get("Solution")
    if Solution is None:
        return {"status": "no_marker", "value": "no Solution class defined"}

    try:
        sol = Solution()
        fn = getattr(sol, fn_name, None)
        if fn is None:
            return {"status": "no_marker", "value": f"Solution.{fn_name} missing"}
        result = fn(*args)
        return {"status": "ok", "value": repr(result)[:_MAX_VALUE_LEN]}
    except TimeoutError:
        raise
    except SystemExit as e:
        # Mirror _run_stdin: sys.exit(0) in user code (e.g. trailing __main__
        # block that finishes cleanly) is not a failure; non-zero is run_error.
        if e.code in (0, None):
            return {"status": "ok", "value": repr(None)}
        return {"status": "run_error", "value": f"SystemExit code={e.code}"}
    except Exception as e:
        return {"status": "run_error", "value": f"{type(e).__name__}: {e}"[:300]}


def _run_stdin(code: str, aug_input: str) -> dict:
    """Run code with aug_input piped to sys.stdin, capture sys.stdout.

    Normalizes literal "\\n" → real newline (model-emitted artifact). Then
    delegates to the canonical benchmark harness path in `testing_util` so
    that FMV scoring agrees byte-for-byte with `grade_stdio` on stdio
    submissions. Reuses (no re-implementation):
    - `clean_if_name` — AST-strip `if __name__ == "__main__":` so main runs
    - `make_function` — wrap script into `wrapped_function` for @patch use
    - `compile_code` — exec into ModuleType (timeout=0 here; outer
      `_worker_run_one` SIGALRM already armed for the whole task)
    - `call_method` — invoke under @patch decorators for `sys.stdin`,
      `sys.stdin.readline`, `.readlines`, `.read`, and `builtins.open`
    - `Capturing` — context manager that swaps `sys.stdout` to a StringIO
    """
    s = (aug_input or "")
    s = s.replace("\\\\n", "\n").replace("\\n", "\n")
    s = s.replace("\\\\t", "\t").replace("\\t", "\t")

    from types import ModuleType

    from recipe.code_bench.testing_util import (
        Capturing, call_method, clean_if_name, make_function,
    )

    try:
        wrapped_code = make_function(clean_if_name(code))
    except TimeoutError:
        raise
    except Exception as e:
        return {"status": "run_error", "value": f"wrap_error: {type(e).__name__}: {e}"[:300]}

    # Inline exec into a fresh ModuleType. We intentionally do NOT call
    # testing_util.compile_code here because it does `signal.alarm(timeout)` +
    # `signal.alarm(0)` internally, which would cancel the outer SIGALRM
    # armed by `_worker_run_one` for the whole task — leaving user code
    # (e.g. `while True: pass`) unbounded and hanging the worker. Inline
    # exec without nested alarm preserves the outer timeout guarantee.
    tmp_sol = ModuleType("tmp_sol", "")
    try:
        exec(wrapped_code, tmp_sol.__dict__)
    except TimeoutError:
        raise
    except Exception as e:
        return {"status": "run_error", "value": f"compile_error: {type(e).__name__}: {e}"[:300]}

    method = getattr(tmp_sol, "wrapped_function", None)
    if method is None:
        return {"status": "no_marker", "value": "no wrapped_function after make_function"}

    try:
        with Capturing() as captured:
            call_method(method, s)
    except TimeoutError:
        raise
    except Exception as e:
        return {"status": "run_error", "value": f"{type(e).__name__}: {e}"[:300]}

    out = (captured[0] if captured else "").strip()
    return {"status": "ok", "value": out[:_MAX_VALUE_LEN]}


# ---- Per-candidate subprocess worker (runs in spawned mp.Process) ----

def _run_candidate_in_subprocess(
    code: str,
    aug_inputs: list[str],
    fn_name: str | None,
    per_input_timeout: int,
    conn,
) -> None:
    """Subprocess entrypoint: init preamble + reliability_guard, run all K inputs,
    send list[K] of {status, value} back over Pipe. Never returns to caller (the
    subprocess exits when this function returns or is SIGKILLed by main watchdog).

    Each input goes through `_worker_run_one`, which sets its own per-input
    SIGALRM (cheap inner-defense for normal timeouts). If SIGALRM is masked by
    candidate code (e.g. @unittest.mock.patch chain), this whole subprocess
    hangs — that is exactly when the parent's watchdog SIGKILLs us.
    """
    try:
        _worker_init()
        outputs = [_worker_run_one(code, inp, fn_name, per_input_timeout) for inp in aug_inputs]
        conn.send(outputs)
    except BaseException as e:  # noqa: BLE001 - subprocess boundary, must report all errors
        # Bare except: this is the OUTER boundary of a spawned subprocess. If we
        # let an exception propagate, the parent gets no result and infers
        # crash from exitcode — losing diagnostic detail. Send a structured
        # error per input instead.
        try:
            conn.send([
                {"status": "run_error", "value": f"subprocess_error: {type(e).__name__}: {e}"[:300]}
                for _ in aug_inputs
            ])
        except Exception:
            # Best-effort error report from an already-failing subprocess. We
            # are inside the outer BaseException handler — any further failure
            # here (broken pipe, pickling error on weird exception payloads,
            # SIGKILL race) is swallowed because raising would just turn into
            # exitcode=1 with no diagnostic. Broad catch is intentional.
            pass
    finally:
        try:
            conn.close()
        except Exception:
            # Resource cleanup; never mask original exception.
            pass


# ---- Public API (main process) ----

def _capture_one_candidate_input(
    code: str,
    aug_input: str,
    fn_name: str | None,
    per_input_timeout: int,
) -> dict:
    """Spawn 1 fresh subprocess for one (candidate, input) execution.

    This is the pair-level counterpart of `_capture_one_candidate`. It keeps
    the same inner SIGALRM timeout but uses a per-input outer watchdog instead
    of the candidate-level K * timeout watchdog, which avoids one slow private
    input blocking all remaining inputs for that candidate.
    """
    if not (code or "").strip():
        return {"status": "no_code", "value": ""}

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_run_candidate_in_subprocess,
        args=(code, [aug_input], fn_name, per_input_timeout, child_conn),
    )
    proc.start()
    child_conn.close()

    watchdog_timeout = max(1, int(per_input_timeout)) + _PAIR_WATCHDOG_SLACK_SECONDS
    poll_interval = 0.1
    start = time.time()
    killed = False
    try:
        while proc.is_alive():
            if time.time() - start > watchdog_timeout:
                proc.kill()
                killed = True
                break
            time.sleep(poll_interval)
        proc.join(2)

        result: dict | None = None
        if parent_conn.poll(2.0):
            try:
                payload = parent_conn.recv()
                if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
                    result = payload[0]
            except (EOFError, OSError, pickle.UnpicklingError):
                pass

        if result is not None:
            return result

        if killed:
            return {"status": "timeout", "value": "candidate_input_killed_by_watchdog"}

        ec = proc.exitcode
        return {"status": "run_error", "value": f"subprocess_exited_exitcode={ec}"}
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        try:
            parent_conn.close()
        except Exception:
            pass


def _capture_outputs_batch_by_candidate(
    code_list: list[str],
    aug_inputs: list[str],
    fn_name: str | None,
    timeout: int,
    max_workers: int,
) -> list[list[dict]]:
    N = len(code_list)
    K = len(aug_inputs)
    results: list[list[dict]] = [[{"status": "missing", "value": ""} for _ in range(K)] for _ in range(N)]
    if N == 0 or K == 0:
        return results

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as tpool:
        futures = {
            tpool.submit(_capture_one_candidate, code, aug_inputs, fn_name, timeout): i
            for i, code in enumerate(code_list)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[i] = [
                    {"status": "run_error", "value": f"orchestrator_error: {type(e).__name__}: {e}"[:300]}
                    for _ in range(K)
                ]

    return results


def _capture_outputs_batch_by_pair(
    code_list: list[str],
    aug_inputs: list[str],
    fn_name: str | None,
    timeout: int,
    max_workers: int,
) -> list[list[dict]]:
    N = len(code_list)
    K = len(aug_inputs)
    results: list[list[dict]] = [[{"status": "missing", "value": ""} for _ in range(K)] for _ in range(N)]
    if N == 0 or K == 0:
        return results

    # Schedule longer inputs first. This preserves deterministic results while
    # starting likely stress-test tails early instead of leaving them at the end
    # of a candidate-local sequential queue.
    tasks = [(i, k) for i in range(N) for k in range(K)]
    tasks.sort(key=lambda item: len(aug_inputs[item[1]] or ""), reverse=True)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as tpool:
        futures = {
            tpool.submit(
                _capture_one_candidate_input,
                code_list[i],
                aug_inputs[k],
                fn_name,
                timeout,
            ): (i, k)
            for i, k in tasks
        }
        for fut in as_completed(futures):
            i, k = futures[fut]
            try:
                results[i][k] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[i][k] = {
                    "status": "run_error",
                    "value": f"orchestrator_error: {type(e).__name__}: {e}"[:300],
                }

    return results


def _capture_one_candidate(
    code: str,
    aug_inputs: list[str],
    fn_name: str | None,
    per_input_timeout: int,
) -> list[dict]:
    """Spawn 1 fresh subprocess for this candidate, watchdog-poll, SIGKILL on
    deadline. Returns list[K] of {status, value} matching aug_inputs.

    Per-candidate watchdog deadline = K * (per_input_timeout + 1) + 5, mirroring
    `recipe/code_bench/reward_score.py:check_correctness:206`. Floor at 60s so
    very small batches still have room for subprocess startup.
    """
    K = len(aug_inputs)
    if not (code or "").strip():
        return [{"status": "no_code", "value": ""} for _ in range(K)]
    if K == 0:
        return []

    # Use fork (Linux default), mirroring `reward_score.py:check_correctness`
    # which has been running stably in Ray actors for months. Earlier iteration
    # used spawn here, but spawn re-executes the parent's __main__ via
    # runpy.run_path — inside a Ray worker this is `default_worker.py`, which
    # triggers `import ray → asyncio` chain in partially-initialized state and
    # raises ImportError(base_futures circular import), silently corrupting FMV
    # scores. Fork COW-copies parent memory, no re-import needed.
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_run_candidate_in_subprocess,
        args=(code, aug_inputs, fn_name, per_input_timeout, child_conn),
    )
    proc.start()
    # Parent does not write to the pipe; close our end of the child handle so
    # EOF semantics work correctly on parent_conn.poll() / recv().
    child_conn.close()

    candidate_timeout = max(60, K * (per_input_timeout + 1) + 5)
    poll_interval = 0.1
    start = time.time()
    killed = False
    try:
        while proc.is_alive():
            if time.time() - start > candidate_timeout:
                # OS-level SIGKILL — cannot be bypassed by any user code,
                # unlike the in-process SIGALRM inside _worker_run_one.
                proc.kill()
                killed = True
                break
            time.sleep(poll_interval)
        proc.join(2)

        # Try to drain result from pipe (subprocess may have sent before exit
        # even in the killed-mid-input case — partial K is still useful).
        # Use a generous poll budget (2s) so a slow writer flushing a
        # multi-KB payload right at the deadline boundary is not lost. After
        # the child exits / is killed, the pipe's writer end is closed and
        # poll() returns immediately once there is data or EOF, so the
        # budget is an upper bound, not a per-call cost.
        result: list[dict] | None = None
        if parent_conn.poll(2.0):
            try:
                payload = parent_conn.recv()
                if isinstance(payload, list) and len(payload) == K:
                    result = payload
            except (EOFError, OSError, pickle.UnpicklingError):
                # EOFError: child closed pipe before sending full pickle.
                # OSError: pipe broken (rare race).
                # UnpicklingError: SIGKILL truncated mid-pickle.
                # All three mean "no usable payload"; fall through to the
                # killed / exited branches below.
                pass

        if result is not None:
            return result

        if killed:
            return [{"status": "timeout", "value": "candidate_killed_by_watchdog"} for _ in range(K)]

        # Subprocess exited without sending a result and was not killed by us:
        # either it crashed (segfault, OOM, sys.exit at module level) or the
        # pipe send raced. Surface as run_error with exitcode for diagnostic.
        ec = proc.exitcode
        return [{"status": "run_error", "value": f"subprocess_exited_exitcode={ec}"} for _ in range(K)]
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        try:
            parent_conn.close()
        except Exception:
            # Resource cleanup in finally; never let close-side failures mask
            # the real result/error returned above.
            pass


def capture_one_output(code: str, aug_input: str, fn_name: str | None, timeout: int) -> dict:
    """Execute `code` against a single `aug_input`. Thin wrapper over
    `_capture_one_candidate` — kept as part of the module's public API for
    callers that want one-shot scoring.

    Dispatches by problem type:
    - fn_name non-empty → functional (LeetCode-style Solution().fn_name(*args))
    - fn_name None/empty → STDIN (aug_input piped to stdin, capture stdout)
    """
    return _capture_one_candidate(code, [aug_input], fn_name, timeout)[0]


def capture_outputs_batch(
    code_list: list[str],
    aug_inputs: list[str],
    fn_name: str | None,
    timeout: int,
    max_workers: int = 16,
    granularity: str = "candidate",
) -> list[list[dict]]:
    """Execute N candidates × K aug inputs.

    Returns [N][K] of {status, value}. fn_name None/empty → STDIN dispatch.

    `max_workers` caps concurrent subprocesses. `granularity="candidate"`:
    one subprocess runs all K inputs for a candidate sequentially under a
    candidate-level watchdog. `granularity="pair"`: each (candidate, input)
    runs in its own subprocess with a per-input watchdog, so one hanging or
    crashing input cannot wipe out the candidate's other K-1 outputs, at the
    cost of more fork overhead. Both use the same per-input SIGALRM timeout.
    """
    granularity_norm = (granularity or "candidate").strip().lower().replace("-", "_")
    if granularity_norm in {"candidate", "candidate_level"}:
        return _capture_outputs_batch_by_candidate(code_list, aug_inputs, fn_name, timeout, max_workers)
    if granularity_norm in {"pair", "pairwise", "candidate_input", "input"}:
        return _capture_outputs_batch_by_pair(code_list, aug_inputs, fn_name, timeout, max_workers)
    raise ValueError(
        f"Unsupported FMV execution granularity {granularity!r}; expected 'candidate' or 'pair'."
    )
