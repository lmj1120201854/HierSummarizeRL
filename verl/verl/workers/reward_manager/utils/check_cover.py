import json
import numpy as np
import time

from verl.workers.reward_manager.utils.prompts import coverage_prompt_template
from tqdm import tqdm

from verl.workers.reward_manager.utils.apis import request_cover_check
import concurrent.futures
from json_repair import repair_json


def get_score(eval_result):
    if not isinstance(eval_result, list):
        return 0.0, 0.0

    if eval_result == []:
        return 0.0, 0.0
    
    if isinstance(eval_result[0], list):
        eval_result = eval_result[0]
    
    result = []
    for item in eval_result:
        try:
            conclusion = item.get("conclusion", "").replace("**", "").strip()
        except:
            conclusion = "未覆盖"

        if conclusion in ["覆盖", "部分覆盖", "未覆盖"]:
            result.append(conclusion)
        else:
            result.append("未覆盖")
    
    try:
        recall_score = (result.count("覆盖") + result.count("部分覆盖") * 0.5) / len(result)
        precison_score = (result.count("覆盖") + result.count("部分覆盖") * 0.5) / (result.count("覆盖") + result.count("部分覆盖"))
        return recall_score, precison_score
    except:
        return 0.0, 0.0


def get_scores(item):
    ex_score = get_score(item["extreme_short_eval"])
    short_score = get_score(item["short_eval"])
    long_score = get_score(item["long_eval"])
    return ex_score, short_score, long_score


def parse_eval_response(eval_response):
    try:
        eval_response = eval_response.replace("```json", "").replace("```", "").strip()
        good_json_string = repair_json(eval_response)
        eval_result = json.loads(good_json_string)
        return eval_result
    except:
        return []


def parse_res_response(response):
    try:
        eval_result = response.replace("```json", "").replace("```", "").strip()
        good_json_string = repair_json(eval_result)
        eval_result = json.loads(good_json_string)
        # check key
        if "extreme_short" not in eval_result:
            eval_result["extreme_short"] = ""
        
        if "short" not in eval_result:
            eval_result["short"] = ""

        if "long" not in eval_result:
            eval_result["long"] = ""

        return eval_result
    except Exception as e:
        return {"extreme_short": "", "short": "", "long": ""}


def get_key_points_str(strs):
    text = ""
    for idx, g in enumerate(strs):
        text += f"{str(idx+1)}. {g}"
        if idx != len(strs) - 1:
            text += "\n"
    
    return text

    
def process(item):
    # 1. 提取response
    response = parse_res_response(item["response"])

    # 2. 三级分别进行提取
    ex_prompt = coverage_prompt_template.format(summary=response["extreme_short"], key_points=get_key_points_str(item["summary_points"]["extreme_short_points"]))
    
    short_prompt = coverage_prompt_template.format(summary=response["short"], key_points=get_key_points_str(item["summary_points"]["short_points"]))

    long_prompt = coverage_prompt_template.format(summary=response["long"], key_points=get_key_points_str(item["summary_points"]["long_points"]))

    # 开并发加速
    prompts = [ex_prompt, short_prompt, long_prompt]
    max_workers = 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(request_cover_check, prompt) for prompt in prompts]
        eval_responses = [future.result() for future in futures]
    
    # 
    ex_eval_response, short_eval_response, long_eval_response = eval_responses[0], eval_responses[1], eval_responses[2]
    ex_eval = parse_eval_response(ex_eval_response)
    short_eval = parse_eval_response(short_eval_response)
    long_eval = parse_eval_response(long_eval_response)

    return ex_eval, short_eval, long_eval


def process_eval(data):
    max_workers = 128
    futures = []
    result = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in data:
            futures.append(executor.submit(process, item))

        pairs = [(item, future) for item, future in zip(data, futures)]
        print(f"开始抓取覆盖率评估, 数量{len(data)}, 并发数{max_workers}")
        for item, future in tqdm(pairs):
            points = future.result()
            ex, short, _long = points
            new_item = {
                "extreme_short_eval": ex,
                "short_eval": short, 
                "long_eval": _long
            }
            result.append(new_item)
    return result


def get_cover_scores(data):
    result = process_eval(data)
    scores = [get_scores(item) for item in result]
    return scores
