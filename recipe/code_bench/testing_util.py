"""
Run LiveCodeBench-style code execution without the `pyext` dependency.

Use this module when evaluating generated code for call-based or standard-input
problems in the recipe-local LiveCodeBench pipeline.
"""

import ast
import faulthandler
import json
import platform
import signal
import sys
import time
from datetime import datetime
from decimal import Decimal
from enum import Enum
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def truncatefn(value, length=300):
    if not isinstance(value, str):
        value = str(value)
    if len(value) <= length:
        return value
    return value[: length // 2] + "...(truncated) ..." + value[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("timeout occured: alarm went off")
    raise TimeoutException


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        self._stringio = StringIO()
        self._stringio.close = lambda _ignored=None: None
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        return import_string + "\n" + ast.unparse(import_stmts) + "\n" + ast.unparse(function_ast)
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            return None

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    finally:
        signal.alarm(0)
    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def get_stripped_lines(value: str):
    value = value.strip()
    return [value_line.strip() for value_line in value.split("\n")]


def grade_call_based(
    code: str,
    all_inputs: list,
    all_outputs: list,
    fn_name: str,
    timeout: int,
    complete_evaluation: bool = False,
):
    raw_code = code
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return None

    method = get_function(compiled_sol, fn_name)
    if method is None:
        return None

    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    error_info = None

    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out
            all_results.append(tmp_result)

            if not tmp_result and not complete_evaluation:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                    "code": raw_code,
                }
            if not tmp_result and error_info is None:
                error_info = {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                    "code": raw_code,
                }
        except Exception as exc:
            signal.alarm(0)
            error_code = -3 if "timeoutexception" in repr(exc).lower() else -4
            current_error = {
                "error": repr(exc),
                "error_code": error_code,
                "error_message": "Time Limit Exceeded" if error_code == -3 else "Runtime Error",
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gt_out),
                "code": raw_code,
            }
            all_results.append(error_code)
            if not complete_evaluation:
                return all_results, current_error
            if error_info is None:
                error_info = current_error
        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, error_info if error_info else {"execution time": total_execution, "code": raw_code}


def grade_stdio(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
    complete_evaluation: bool = False,
):
    raw_code = code
    code = clean_if_name(code)
    code = make_function(code)

    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return None

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return None

    all_results = []
    total_execution_time = 0
    error_info = None

    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()

        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as exc:
                signal.alarm(0)
                error_code = -3 if "timeoutexception" in repr(exc).lower() else -4
                current_error = {
                    "error": repr(exc),
                    "error_code": error_code,
                    "error_message": "Time Limit Exceeded" if error_code == -3 else "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "code": raw_code,
                }
                all_results.append(error_code)
                if not complete_evaluation:
                    return all_results, current_error
                if error_info is None:
                    error_info = current_error
                continue
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]
        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)
        wrong_answer_payload = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
            "code": raw_code,
        }

        has_wrong_answer = False
        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            has_wrong_answer = True
            wrong_answer_payload["error_message"] = "Wrong answer: mismatched output length"
        else:
            for output_line_idx, (prediction_line, gt_line) in enumerate(
                zip(stripped_prediction_lines, stripped_gt_out_lines)
            ):
                wrong_answer_payload["error_message"] = (
                    f"Wrong answer at output_line_idx={output_line_idx}: "
                    f"{truncatefn(prediction_line)} != {truncatefn(gt_line)}"
                )
                if prediction_line == gt_line:
                    continue

                prediction_ok, prediction_decimals = convert_line_to_decimals(prediction_line)
                if not prediction_ok:
                    has_wrong_answer = True
                    break

                gt_ok, gt_decimals = convert_line_to_decimals(gt_line)
                if not gt_ok:
                    has_wrong_answer = True
                    break

                if prediction_decimals == gt_decimals:
                    continue

                has_wrong_answer = True
                break

        if has_wrong_answer:
            all_results.append(-2)
            if not complete_evaluation:
                return all_results, wrong_answer_payload
            if error_info is None:
                error_info = wrong_answer_payload
        else:
            all_results.append(True)

    return all_results, error_info if error_info else {"execution time": total_execution_time, "code": raw_code}


def run_test(sample, test=None, debug=False, timeout=6, complete_evaluation=False):
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    in_outs = json.loads(sample["input_output"])
    if in_outs.get("fn_name") is None:
        which_type = CODE_TYPE.standard_input
        method_name = None
    else:
        which_type = CODE_TYPE.call_based
        method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        raise AssertionError("should not happen: test code is none")

    num_tests = len(in_outs["inputs"])
    failure_results = [-4 for _ in range(num_tests)]

    if which_type == CODE_TYPE.call_based:
        signal.alarm(timeout)
        try:
            results, metadata = grade_call_based(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                fn_name=method_name,
                timeout=timeout,
                complete_evaluation=complete_evaluation,
            )
            return results, metadata
        except Exception as exc:
            return failure_results, {"error_code": -4, "error_message": f"Error during testing: {exc}"}
        finally:
            signal.alarm(0)

    signal.alarm(timeout)
    try:
        results, metadata = grade_stdio(
            code=test,
            all_inputs=in_outs["inputs"],
            all_outputs=in_outs["outputs"],
            timeout=timeout,
            complete_evaluation=complete_evaluation,
        )
        return results, metadata
    except Exception as exc:
        return failure_results, {"error_code": -4, "error_message": f"Error during testing: {exc}"}
    finally:
        signal.alarm(0)


def reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    import os
    import shutil
    import subprocess

    os.environ["OMP_NUM_THREADS"] = "1"

    builtins.quit = None
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    # Stub raising OSError instead of None: numpy's import-time CPU probe
    # (check_support_sve) calls subprocess.run and only catches OSError, so a
    # None Popen makes any `import scipy/sklearn` crash with TypeError.
    def _blocked_popen(*args, **kwargs):
        raise OSError("subprocess is disabled by the execution sandbox")

    subprocess.Popen = _blocked_popen  # type: ignore

    __builtins__["help"] = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
