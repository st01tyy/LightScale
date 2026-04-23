import json
import re
import os
from typing import Tuple


def extract_tools_for_cm(text: str) -> list:
    tool_call_start_prefix = "Action:"
    tool_call_args_start_prefix = "\nAction Input:"
    code_interpreter_tool_name = "code_interpreter"
    code_content_key = "code_content"
    tool_calls = []
    while True:
        i = text.rfind(tool_call_start_prefix)
        j = text.rfind(tool_call_args_start_prefix)
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            func_name = text[i + len(tool_call_start_prefix) : j].strip()
            func_args = text[j + len(tool_call_args_start_prefix) :].strip()
            if func_name == code_interpreter_tool_name:
                func_args, valid = convert_to_json_str(func_args)
                if not valid:
                    if not func_args.startswith("```py\n"):
                        func_args = "```py\n" + func_args
                    if not func_args.endswith("\n```"):
                        func_args = func_args + "\n```"
                    func_args = json.dumps({code_content_key: func_args}, ensure_ascii=False)
            else:
                func_args = convert_to_json_str(func_args)[0]
            try:
                func_args_dict = json.loads(func_args)
                tool_calls.insert(0, {"name": func_name, "arguments": func_args_dict})
            except json.JSONDecodeError:
                pass
            text = text[:i]
        else:
            break
    return tool_calls


def convert_to_json_str(input_str: str):
    try:
        json.loads(input_str)
        return input_str, True
    except Exception as e:
        try:
            json_object = eval(input_str)
            return json.dumps(json_object, ensure_ascii=False), True
        except Exception as e:
            try:
                value = input_str.replace("'", '"')
                json.loads(value)
                return value, True
            except Exception as e:
                pass
    return input_str, False


def extract_tools_for_qw(text: str) -> list:
    """
    尝试抽取qwen返回的tools结果，例如：
    '{"name": "enterprise_app_lawfirmdetailbyname_query_php", "arguments": {"keyword": "北京市中伦律师事务所"}}\n{"name": "enterprise_app_lawfirmdetailbyname_query_php", "arguments": {"keyword": "987654321098765432"}}'
    """
    # 匹配所有可能存在的JSON字符串
    json_strings = []
    for line in text.split("\n"):
        # 在每行中查找符合JSON格式的部分
        matches = re.findall(r"\{.*?\}", line, re.DOTALL)
        for match in matches:
            try:
                # 验证是否为有效JSON
                json.loads(match)
                json_strings.append(match)
            except json.JSONDecodeError:
                continue

    # 更精准的二次匹配（处理嵌套结构）
    # 若上述方法不奏效，可尝试逐字符扫描
    if not json_strings:
        json_strings = []
        stack = []
        start_index = -1
        for i, char in enumerate(text):
            if char == "{":
                stack.append(char)
                if start_index == -1:
                    start_index = i
            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        json_candidate = text[start_index : i + 1]
                        try:
                            json.loads(json_candidate)
                            json_strings.append(json_candidate)
                        except json.JSONDecodeError:
                            pass
                        start_index = -1

    return [json.loads(s) for s in json_strings]


def cal_tool_calls_reward_score(correct_dict: dict, assessed_dict: dict) -> float:
    """
    评估两个tool_call的相似度，返回0-1之间的评分
    结构化字典评分程序（针对 {name, arguments} 格式优化）
    评分逻辑：
    - 若name字段不匹配直接返回0分
    - arguments部分按缺失键、多余键、值错误分别扣分
    - 最终得分 = 1 - 总扣分（扣分上限为1）
    """

    # 阶段一：校验核心字段 --------------------------------------------------
    if correct_dict.get("name") != assessed_dict.get("name"):
        return 0.0  # 名称不匹配直接0分
    if not isinstance(assessed_dict.get("arguments", ""), dict):
        return 0.0  # arguments不存在或者不是字典类型直接0分

    # 阶段二：比较arguments结构 ----------------------------------------------
    def compare_arguments(correct_args, assessed_args):
        """递归比较arguments结构，返回差异统计"""
        diffs = {"missing": 0, "extra": 0, "value": 0}
        # 键集合比较
        correct_keys = set(correct_args.keys())
        assessed_keys = set(assessed_args.keys())
        diffs["missing"] = len(correct_keys - assessed_keys)
        diffs["extra"] = len(assessed_keys - correct_keys)

        # 共有键值比较
        for key in correct_keys & assessed_keys:
            c_val = correct_args[key]
            a_val = assessed_args[key]

            # 递归处理嵌套字典
            if isinstance(c_val, dict) and isinstance(a_val, dict):
                nested_diffs = compare_arguments(c_val, a_val)
                for k in diffs:
                    diffs[k] += nested_diffs[k]
            # 处理列表类型（例如参数值为列表）
            elif isinstance(c_val, list) and isinstance(a_val, list):
                if len(c_val) != len(a_val):
                    diffs["value"] += 1  # 列表长度不同视为值错误
                else:
                    for c_item, a_item in zip(c_val, a_val):
                        if c_item != a_item:
                            diffs["value"] += 1
            # 普通值比较
            elif c_val != a_val:
                diffs["value"] += 1
        return diffs

    # 执行比较
    args_diffs = compare_arguments(correct_dict.get("arguments", {}), assessed_dict.get("arguments", {}))

    # 阶段三：动态权重扣分 --------------------------------------------------
    # 权重配置（可根据业务需求调整）
    penalty_weights = {
        "missing": 0.5,  # 缺失关键参数权重最高
        "extra": 0.2,  # 多余参数权重中等
        "value": 0.3,  # 值错误权重根据关键性调整
    }

    # 计算总扣分（加权求和）
    total_penalty = sum(args_diffs[err_type] * penalty_weights[err_type] for err_type in args_diffs)

    # 扣分上限保护（避免负分）
    final_score = max(0.0, 1.0 - min(total_penalty, 1.0))
    return round(final_score, 2)


def cal_tool_calls_reward_score_v2(excepted_dicts: list, extracted_dicts: list) -> float:
    """
    根据《ToolRL: Reward is All Tool Learning Needs》，将reward分为3个部分：
    1. name
    2. params
    3. value
    返回分数在[-3,3]
    """
    matched_tool_name = matched_param = matched_value = 0
    tool_name_score = param_score = value_score = 0
    all_tool_name = all_param = 0
    for i, excepted_tool in enumerate(excepted_dicts):
        all_tool_name += 1
        excepted_keys = set(excepted_tool.get("arguments", {}).keys())
        all_param += len(excepted_keys)

        if i > len(extracted_dicts) - 1:
            continue

        # 计算工具名称匹配分数
        extracted_tool = extracted_dicts[i]
        if extracted_tool.get("name") == excepted_tool.get("name"):
            matched_tool_name += 1

        # 计算参数key匹配分数
        if isinstance(extracted_tool.get("arguments", ""), dict):
            extracted_keys = set(extracted_tool.get("arguments", {}).keys())
            if len(excepted_keys | extracted_keys) == 0:
                param_score += 1
            else:
                param_score += len(excepted_keys & extracted_keys) / len(excepted_keys | extracted_keys)
            # 计算参数value匹配分数
            for key in excepted_keys & extracted_keys:
                if excepted_tool["arguments"][key] == extracted_tool.get("arguments", {})[key]:
                    value_score += 1
    all_score = matched_tool_name / all_tool_name + param_score + value_score
    max_score = 1 + all_tool_name + all_param
    return 6 * all_score / max_score - 3


def verify_tool_calls_for_qw(
    raw_answer: str, ground_truth: str, tool_call_start="<tool_call>", tool_call_end="</tool_call>"
) -> bool:
    excepted_tools = [
        {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
        for t in json.loads(ground_truth).get("tool_calls", [])
    ]
    # 此时，真实情况不需要调用
    if len(excepted_tools) == 0:
        # 一旦出现工具调用token，则不给分
        if tool_call_start in raw_answer or tool_call_end in raw_answer:
            return False
        return True
    tool_call_pattern = rf"{tool_call_start}(.*?){tool_call_end}"

    # 抽取tools
    all_extracted_tools = []
    tool_call_raw_matches = re.findall(tool_call_pattern, raw_answer, re.DOTALL)
    # 如果没找到tool token标签则返回False
    if not tool_call_raw_matches:
        return False
    # 如果找到了，则需要判断工具调用JSON格式是否有效
    is_valid = True
    for tool_call_raw in tool_call_raw_matches:
        extracted_tools = extract_tools_for_qw(tool_call_raw.strip())
        if len(extracted_tools) == 0:
            is_valid = False
            break
        all_extracted_tools.extend(extracted_tools)
    if not is_valid:
        return False

    tool_score, count = 0, 0
    for extracted_tool, excepted_tool in zip(all_extracted_tools, excepted_tools):
        tool_score += cal_tool_calls_reward_score(excepted_tool, extracted_tool)
        count += 1
    if count == 0:
        print("=========get unexcpeted tools=========")
        print("raw_answer: ", raw_answer)
        print("all_extracted_tools: ", all_extracted_tools)
        print("excepted_tools: ", excepted_tools)
        return False
    return tool_score / count >= 0.8


def verify_tool_calls_for_cm(raw_answer: str, ground_truth: str) -> bool:
    # excepted_tools = [
    #     {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
    #     for t in json.loads(ground_truth).get("tool_calls", [])
    # ]
    excepted_tools = extract_tools_for_cm(ground_truth)
    # 此时，真实情况不需要调用
    if len(excepted_tools) == 0:
        print("no excepted tools in ground truth")
        print("ground_truth: ", ground_truth)
        print("raw_answer: ", raw_answer)
        # 一旦出现工具调用token，则不给分
        if "Action:" in raw_answer or "Action Input:" in raw_answer:
            return False
        return True

    if not raw_answer.startswith("Action:"):
        return False
    all_extracted_tools = extract_tools_for_cm(raw_answer)
    if len(all_extracted_tools) == 0:
        print("no extracted tools in raw answer")
        print("raw_answer: ", raw_answer)
        print("ground_truth: ", ground_truth)
        return False

    tool_score, count = 0, 0
    for extracted_tool, excepted_tool in zip(all_extracted_tools, excepted_tools):
        tool_score += cal_tool_calls_reward_score(excepted_tool, extracted_tool)
        count += 1
    if count == 0:
        print("=========get unexcpeted tools=========")
        print("raw_answer: ", raw_answer)
        print("all_extracted_tools: ", all_extracted_tools)
        print("excepted_tools: ", excepted_tools)
        return False
    return tool_score / count >= 0.8


def verify_tool_calls_for_cm_v2(raw_answer: str, ground_truth: str) -> Tuple[bool, float]:
    """
    参考论文《ToolRL: Reward is All Tool Learning Needs》
    """
    excepted_tools = extract_tools_for_cm(ground_truth)
    # 此时，真实情况不需要调用
    if len(excepted_tools) == 0:
        print("no excepted tools in ground truth")
        print("ground_truth: ", ground_truth)
        print("raw_answer: ", raw_answer)
        # 一旦出现工具调用token，则不给分
        if "Action:" in raw_answer or "Action Input:" in raw_answer:
            return True, -3
        return True, 3

    # 如果除think、action之外的，还包含其他的内容，返回-3
    if not raw_answer.startswith("Action:"):
        return True, -3

    all_extracted_tools = extract_tools_for_cm(raw_answer)
    if len(all_extracted_tools) == 0:
        print("no extracted tools in raw answer")
        print("raw_answer: ", raw_answer)
        print("ground_truth: ", ground_truth)
        return True, -3

    score = cal_tool_calls_reward_score_v2(excepted_tools, all_extracted_tools)
    return True, round(score, 3)


if __name__ == "__main__":
    # text1 = """
    # 这是一个示例文本，其中包含多个 <tool_call>'{"name": "func1","arguments":{"k11":"v","k12":"v2"}}'</tool_call>。
    # 这里还有一个
    # <tool_call>
    # '{"name": "func2","arguments":{"k21":"v","k22":"v"}}'
    # </tool_call>。
    # """
    # grount_truth_dict1 = {
    #     "tool_calls": [
    #         {"function": {"name": "func1", "arguments": {"k11": "v", "k12": "v"}}},
    #         {"function": {"name": "func2", "arguments": {"k21": "v", "k22": "v"}}},
    #     ]
    # }
    # ground_truth_str1 = json.dumps(grount_truth_dict1, ensure_ascii=False)
    # print(verify_tool_calls_for_qw(text1, ground_truth_str1))

    # text1 = """
    # 这是一个示例文本，其中包含多个\nAction: func1\nAction Input: \n{"k11":"v","k12":"v2"}
    # Action: func2\nAction Input: {"k21":"v","k22":"v"}
    # """
    text1 = """Action: func1\nAction Input: \n{"k11":"v","k12":{"k":"v","k2":1,"k3":{"v":0}}}
    Action: func2\nAction Input: {"k21":"v","k22":"v"}
    """
    grount_truth_dict1 = {
        "tool_calls": [
            {"function": {"name": "func1", "arguments": {"k11": "v", "k12": "v", "k13": 1}}},
            {"function": {"name": "func2", "arguments": {"k21": "v", "k22": "v"}}},
        ]
    }
    ground_truth_str1 = 'Action: func1\nAction Input: {"k11": "v", "k12":{"k":"v","k2":1,"k3":{"v":0}},"k13": 1}\nAction: func2\nAction Input: {"k21": "v", "k22": "v"}\nAction: func3\nAction Input: {"k31": "v", "k32": "v"}'
    print(verify_tool_calls_for_cm_v2(text1, ground_truth_str1))
