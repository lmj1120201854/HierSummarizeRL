import json


def calculate_json_format_reward(model_output):
    cleaned_output = model_output.replace("```json", "").replace("```", "").strip()
    try:
        parsed_data = json.loads(cleaned_output)
        if not isinstance(parsed_data, dict):
            return -1
        
        expected_keys = {"extreme_short", "short", "long"}
        actual_keys = set(parsed_data.keys())
        
        if actual_keys != expected_keys:
            return -1
        
        for key in expected_keys:
            if not isinstance(parsed_data.get(key), str):
                return -1
        
        # 所有检查通过
        return 0
        
    except (json.JSONDecodeError, TypeError):
        return -1


def get_token_num(text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    return len(tokens)


def cot_length_reward(length, l_min, l_opt, l_max):
    if length < l_min:
        return -1
    elif length < l_opt:
        return -1 + (length - l_min) / (l_opt - l_min)
    elif length <= l_max:
        return 0
    else:
        return -1


import math

def length_reward(
    response_length, 
    reference_length, 
    tolerance_ratio=0.20, 
    max_penalty=-1.0
):
    if reference_length == 0:
        return max_penalty
    
    lower_bound = reference_length * (1 - tolerance_ratio)
    upper_bound = reference_length * (1 + tolerance_ratio)
    
    if lower_bound <= response_length <= upper_bound:
        return 0.0
    
    # 计算相对偏离比例（从容忍边界开始算起）
    if response_length < lower_bound:
        deviation_ratio = (lower_bound - response_length) / lower_bound
    else:
        deviation_ratio = (response_length - upper_bound) / upper_bound
    
    penalty = max_penalty * deviation_ratio * 3  # 如果偏离10%，则-0.3
    
    # 保证不超过最大惩罚
    return max(penalty, max_penalty)


def calculate_length_reward(
    format_reward, 
    json_format_reward, 
    response, 
    cot,
    groud_truth,
    tokenizer
):
    if format_reward == -1 or json_format_reward == -1:
        return 0  # 不计算reward
    
    cleaned_output = response.replace("```json", "").replace("```", "").strip()
    try:
        parsed_data = json.loads(cleaned_output)
        extreme_short = parsed_data["extreme_short"]
        short = parsed_data["short"]
        _long = parsed_data["long"]
    except:
        return 0

    # 获取长度
    chain_len = get_token_num(cot, tokenizer)

    extreme_len = get_token_num(extreme_short, tokenizer)
    short_len = get_token_num(short, tokenizer)
    long_len = get_token_num(_long, tokenizer)

    # 获取ground_truth长度
    g_extreme_len = get_token_num(groud_truth["extreme_short"], tokenizer)
    g_short_len = get_token_num(groud_truth["short"], tokenizer)
    g_long_len = get_token_num(groud_truth["long"], tokenizer)

    R_total = 0.3 * cot_length_reward(chain_len, 100, 200, 300) + 0.15 * length_reward(extreme_len, g_extreme_len) + 0.25 * length_reward(short_len, g_short_len) + 0.3 * length_reward(long_len, g_long_len)

    return R_total


    
