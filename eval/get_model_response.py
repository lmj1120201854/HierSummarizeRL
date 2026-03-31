import os
import json
import re
import requests
import json
import time
import openai
import concurrent.futures
from tqdm import tqdm
import argparse
from json_repair import repair_json


model_server = os.environ.get('MODEL_SERVER')
model_path = os.environ.get('MODEL_PATH')

prompt_template = """你是一名专业的多级摘要生成助手。请基于提供的标题和正文内容，生成结构化的三级摘要。

## 输入格式
- **标题**：文档标题
- **正文**：需要摘要的完整文本

## 输出要求
### 内容要求
- **极短摘要**：1句话，≤50字，提炼核心要点
- **短摘要**：3-5句话，≤100字，概括关键信息
- **长摘要**：完整概述，≤200字，包含重要细节

### 质量要求
- 保留所有关键事实、人物、事件、数据、结论
- 语言精炼准确，逻辑层次清晰
- 避免重复和冗余信息
- 忠实于原文内容，不添加主观评价

### 格式要求
严格遵循以下JSON结构，且**只能输出JSON格式，不要包含任何其他文字、说明或标记**：
```json
{{
  "extreme_short": "极短摘要内容",
  "short": "短摘要内容", 
  "long": "长摘要内容"
}}
```

现在，请基于以下内容生成三级摘要：

标题：{title}

正文：
{content}

请严格按照上述要求生成JSON格式的三级摘要，且只输出JSON，不要其他任何内容。""".strip()


def calculate_json_format_reward(model_output):
    cleaned_output = model_output.replace("```json", "").replace("```", "").strip()
    cleaned_output = repair_json(cleaned_output)
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


def parse_response(response_str):
    if calculate_json_format_reward(response_str) != 0:
        return {"extreme_short": "", "short": "", "long": ""}
    try:
        eval_result = response_str.replace("```json", "").replace("```", "").strip()
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


def get_response(query):
    client = openai.Client(base_url=f"http://{model_server}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=model_path,
                messages=[{
                    "role": "user",
                    "content": query
                }],
                temperature=0.7,
                max_tokens=4096
            )
            return response.choices[0].message.content.split("</think>")[-1].strip()

        except:
            time.sleep(1)
            continue
    
    return ""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ConVeriRL测评")

    # 添加参数
    parser.add_argument("--model_id", type=str, default="Qwen3-8B", help="模型名")  
    parser.add_argument("--dataset", type=str, default="", help="数据集名称")

    args = parser.parse_args()

    with open(f"./dataset/{args.dataset}", "r") as fin:
        with open(f"./output/{args.dataset}.{args.model_id}.output.jsonl", "w") as fout:
            datas = []
            for line in fin:
                line = line.strip()
                data = json.loads(line)
                datas.append(data)
            
            max_workers = 512
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for data in datas:
                    prompt = prompt_template.format(title=data["title"], content=data["content"])
                    
                    if args.model_id == "Qwen3-8B" or args.model_id == "Qwen3-14B" or "HierSummarizeRL" in args.model_id:
                        futures.append(executor.submit(get_response, prompt))
                    else:
                        raise ValueError("model_id error...")

                pairs = [(data, future) for data, future in zip(datas, futures)]

                for data, future in tqdm(pairs):
                    response = future.result()
                    data["response"] = parse_response(response)
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
