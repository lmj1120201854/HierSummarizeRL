set -x

step=100

local_dir="/HierSummarizeRL-Qwen3-8B-RL-ckpt/global_step_${step}/actor"
hf_path="/HierSummarizeRL-Qwen3-8B-RL-ckpt/global_step_${step}/actor/huggingface"

output_path="/HierSummarizeRL-Qwen3-8B-RL-step${step}-ckpt"

python3 legacy_model_merger.py merge \
    --backend=fsdp \
    --local_dir=$local_dir \
    --hf_model_path=$hf_path \
    --target_dir=$output_path