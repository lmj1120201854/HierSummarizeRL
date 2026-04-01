import json
import numpy as np
import time

from verl.workers.reward_manager.utils.prompts import cf_prompt_template
from tqdm import tqdm

from verl.workers.reward_manager.utils.apis import request_cf_check
import concurrent.futures
from json_repair import repair_json


def get_score(eval_result, key):
    try:
        c_score = eval_result[key]["简洁性"]
        if not isinstance(c_score, float):
            if isinstance(c_score, int):
                c_score = float(c_score)
            elif isinstance(c_score, list):
                try:
                    c_score = float(c_score[0])
                except:
                    c_score = 0.0
            elif isinstance(c_score, str):
                try:
                    c_score = float(c_score.strip())
                except:
                    c_score = 0.0
            else:
                c_score = 0.0
    except:
        c_score = 0.0
    
    try:
        f_score = eval_result[key]["流畅性"]
        if not isinstance(f_score, float):
            if isinstance(f_score, int):
                f_score = float(f_score)
            elif isinstance(f_score, list):
                try:
                    f_score = float(f_score[0])
                except:
                    f_score = 0.0
            elif isinstance(f_score, str):
                try:
                    f_score = float(f_score.strip())
                except:
                    f_score = 0.0
            else:
                f_score = 0.0
    except:
        f_score = 0.0
    
    if c_score < 0:
        c_score = 0
    if c_score > 1:
        c_score = 1.0
        
    if f_score < 0:
        f_score = 0
    if f_score > 1:
        f_score = 1.0
    
    return 0.5 * c_score + 0.5 * f_score


def get_scores(eval_result):
    # ex_short
    ex_score = get_score(eval_result, "极短摘要")
    short_score = get_score(eval_result, "短摘要")
    long_score = get_score(eval_result, "长摘要")

    return 0.2 * ex_score + 0.3 * short_score + 0.5 * long_score


def parse_eval_response(eval_response):
    try:
        eval_response = eval_response.replace("```json", "").replace("```", "").strip()
        good_json_string = repair_json(eval_response)
        eval_result = json.loads(good_json_string)
        return eval_result
    except:
        # 处理异常
        eval_result = {
            "极短摘要": {
                "简洁性": 0.0,
                "流畅性": 0.0
            },
            "短摘要": {
                "简洁性": 0.0,
                "流畅性": 0.0
            },
            "长摘要": {
                "简洁性": 0.0,
                "流畅性": 0.0
            }
        }
        return eval_result


def parse_res_response(response):
    try:
        eval_result = response.replace("```json", "").replace("```", "").strip()
        good_json_string = repair_json(eval_result)
        eval_result = json.loads(good_json_string)
        # 三个是否在里面
        if "extreme_short" not in eval_result:
            eval_result["extreme_short"] = ""
        
        if "short" not in eval_result:
            eval_result["short"] = ""

        if "long" not in eval_result:
            eval_result["long"] = ""

        return eval_result
    except Exception as e:
        return {"extreme_short": "", "short": "", "long": ""}

    
def process(item):
    # 1. 提取response
    response = parse_res_response(item["response"])

    # 
    prompt = cf_prompt_template.format(
        title=item["title"],
        content=item["content"],
        extreme_short=response["extreme_short"],
        short=response["short"],
        _long=response["long"]
    )
    eval_response = request_cf_check(prompt)
    eval_result = parse_eval_response(eval_response)

    return eval_result


def process_eval(data):
    max_workers = 128
    futures = []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in data:
            futures.append(executor.submit(process, item))

        pairs = [(item, future) for item, future in zip(data, futures)]
        print(f"开始抓取简洁性、流畅性评估, 数量{len(data)}, 并发数{max_workers}")
        for item, future in tqdm(pairs):
            eval_result = future.result()
            results.append(eval_result)

    return results


def get_cf_scores(data):
    result = process_eval(data)
    scores = [get_scores(item) for item in result]
    return scores
