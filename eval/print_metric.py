import json
import re
import json
import time
from tqdm import tqdm
import argparse

from rouge_chinese import Rouge
import jieba


def get_single_rouge_score(hypothesis, reference):
    hypothesis = ' '.join(jieba.cut(hypothesis))
    reference = ' '.join(jieba.cut(reference))
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores


def get_rouge_scores(hyps, refs):
    hyps = [' '.join(jieba.cut(hyp)) for hyp in hyps]
    refs = [' '.join(jieba.cut(ref)) for ref in refs]
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测评")
    
    # 添加参数
    parser.add_argument("--model_id", type=str, default="Qwen3-8B", help="模型名")  
    parser.add_argument("--dataset", type=str, default="", help="数据集名称")

    args = parser.parse_args()

    datas = []
    with open(f"./output/{args.dataset}.{args.model_id}.output.jsonl", "r") as fin:
        for line in fin:
            line = line.strip()
            data = json.loads(line)
            datas.append(data)
    
    print(f"model name: {args.model_id}")
    
    for key in ["extreme_short", "short", "long"]:
        hyps, refs = [], []
        for data in datas:
            hyp = data["response"][key]
            ref = data["r1_response"][key]
            hyps.append(hyp)
            refs.append(ref)
        
        # print metric
        rouge_score = get_rouge_scores(hyps, refs)
        print(f"---------------------{key}---------------------:")
        print(f"ROUGE SCORE: {rouge_score}")
                