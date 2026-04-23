import logging
import requests
import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed


SUPPORTED_LANGUAGES = ["python", "cpp", "nodejs", "go", "go_test", "java", "php", "csharp", "bash", "typescript", "sql", "rust", "cuda", "lua", "R", "perl", "D_ut", "ruby", "scala", "julia", "pytest", "junit", "kotlin_script", "jest", "verilog", "python_gpu", "lean", "swift", "racket"]

IMPORT_PROMPT = '''from typing import *

from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')

'''

wrapper_code = """
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: {{_raw_input_str[:200]}}\\n")
            return None, True # result, error_occurred

    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method.
            # Errors (e.g., Solution not a class, instantiation fails, method missing)
            # will be caught by the broad except block below.
            _solution_instance = _Solution_class() 
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)
        
        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred

        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        # Serialize result to stdout
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result)) # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
    # Optional: To explicitly exit with an error code if the sandbox relies on it
    # else:
    #    sys.exit(1) 
"""


def query_worker(code_sandbox_url, payload, timeout) -> dict:
    try:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url=code_sandbox_url, data=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"[Error] Request failed for {payload=}\n{e}")
        logging.error(f"Response text: {response.text}")
        return None


def compute_score(code_sandbox_url, completion, test_cases, max_test_cases=10, is_binary_reward=True, timeout=10, max_workers=100) -> float:
    """支持2种类型的标准答案: input-output, test_code
        - input-output: {"inputs": [xxx, yyy], "outputs": [aaa, bbb]}
        - test_code: {"test_code": xxx / [xxx, yyy], "import_prefix": (Optional)}, test_code必须是可直接执行的, 拼到生成的代码后面
    """
    # 提取代码块，如果没找到就0分
    solution = completion
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        # Handle cases like ```\ncode\n```
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    solution = rest
    else:
        logging.debug(f"Missing code block: {completion}")
        return 0.0

    # 检查生成代码语法是否正确
    try:
        tree = ast.parse(solution)
    except Exception as e:
        logging.debug(f"ast.parse fail: {e}")
        logging.debug(f"{completion=}")
        return 0.0

    # 检查测试用例是否合法
    if not isinstance(test_cases, dict):
        try:
            test_cases = json.loads(test_cases)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to json.loads {test_cases=}")
            return 0.0
    if 'inputs' in test_cases and 'outputs' in test_cases:
        if len(test_cases['inputs']) != len(test_cases['outputs']):
            logging.error(f"Mismatch between number of inputs ({len(test_cases['inputs'])}) and outputs ({len(test_cases['outputs'])}).")
            return 0.0
        if len(test_cases['inputs']) == 0:
            logging.error(f"Empty inputs and outputs.")
            return 0.0
        if max_test_cases > 0:
            test_cases['inputs'] = test_cases['inputs'][:max_test_cases]
            test_cases['outputs'] = test_cases['outputs'][:max_test_cases]
    elif 'test_code' in test_cases:
        if len(test_cases['test_code']) == 0:
            logging.error(f"Empty test_code.")
            return 0.0
        if isinstance(test_cases['test_code'], str):
            test_cases['test_code'] = [test_cases['test_code']]
        if max_test_cases > 0:
            test_cases['test_code'] = test_cases['test_code'][:max_test_cases]
    else:
        logging.error("Invalid test_cases structure (missing inputs/outputs, assert_cases, or test_code)")
        return 0.0

    # 填充payload
    payload_list = []
    if "inputs" in test_cases:
        inputs = test_cases['inputs']
        language = test_cases.get('language', 'python')
        fn_name = test_cases.get('fn_name')
        for i, stdin in enumerate(inputs):
            if fn_name and language == 'python':
                final_code = wrapper_code.format(generation=solution, fn_name=fn_name)
            else:
                final_code = solution
            payload = json.dumps({
                'compile_timeout': timeout,
                'run_timeout': timeout,
                'code': final_code,
                'stdin': stdin,
                'language': language,
                'files': {},
                'fetch_files': [],
            })
            payload_list.append((i, payload))
    elif "test_code" in test_cases:
        test_code = test_cases['test_code']
        language = test_cases.get('language', 'python')
        import_prefix = test_cases.get('import_prefix', IMPORT_PROMPT)
        for i, code in enumerate(test_code):
            final_code = f"{import_prefix}\n{solution}\n{code}"
            payload = json.dumps({
                'compile_timeout': timeout,
                'run_timeout': timeout,
                'code': final_code,
                'stdin': '',
                'language': language,
                'files': {},
                'fetch_files': [],
            })
            payload_list.append((i, payload))

    # 多线程访问代码验证器
    request_timeout = timeout + timeout + timeout  # compile_timeout + run_timeout + api_timeout
    thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    future_to_index = {thread_pool.submit(query_worker, code_sandbox_url, payload, request_timeout): i for i, payload in payload_list}

    # 判断返回结果是否正确
    res_list = [False] * len(future_to_index)
    for future in as_completed(future_to_index):
        index = future_to_index[future]
        response = future.result()
        if response is None:
            continue

        status = response.get('status')
        compile_result = response.get("compile_result")
        run_result = response.get("run_result")
        if status == "SandboxError":
            logging.error(f"SandboxError for {code_sandbox_url=}, {response=}")
            continue
        elif status == "Failed":
            logging.debug(f"API returned Failed status.")
            logging.debug(f"Compile Result: {compile_result}\nRun Result: {run_result}")
            continue
        elif status == "Success":
            # Run completed successfully, now check the answer
            if run_result and run_result["status"] == "Finished":
                if "inputs" in test_cases:
                    expected_output = test_cases["outputs"][index]
                    actual_output = run_result.get("stdout")
                    # Note: Output might contain trailing newlines, need normalization
                    if str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                        res_list[index] = True
                elif "test_code" in test_cases:
                    exit_code = run_result.get("return_code")
                    if exit_code == 0:
                        res_list[index] = True
            else:
                # Status is Success but run_result status is not Finished, this is unexpected
                logging.error(f"Status is Success but run_result status is not Finished, this is unexpected.")
                logging.error(f"Compile Result: {compile_result}\nRun Result: {run_result}")
                continue
        else:
            logging.error(f"Unknown api status: {status}")
            logging.error(f"Compile Result: {compile_result}\nRun Result: {run_result}")
            continue

    # 统计正确率
    if is_binary_reward:
        score = float(all(res_list))
    else:
        score = sum(res_list) / len(res_list)
    return score
