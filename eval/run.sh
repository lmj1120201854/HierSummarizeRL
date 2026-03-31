set -x

if [ $# -eq 0 ]; then
  echo "没有输入待评模型ID，请输入模型ID"
  exit 1
fi

model_id=$1

export MODEL_SERVER=127.0.0.1:8000
export MODEL_PATH=HierSummarizeRL-Qwen3-8B-RL-step100-ckpt # 待测评模型

dataset="nlpcc_data_test.jsonl"

python3 get_model_response.py \
--model_id $model_id \
--dataset $dataset

python3 print_metric.py \
--model_id $model_id \
--dataset $dataset