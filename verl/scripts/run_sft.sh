set -x

export WANDB_MODE=online
export WANDB_PROJECT=HierSummarizeRL
export WANDB_ENTITY=lmj1120201854-beijing-institute-of-technology
export WANDB_NAME=SFT

nproc_per_node=4
CONFIG_PATH="/data/home/3120245632/scow/ai/appData/mjli/llm_proj/HierSummarizeRL/verl/verl/trainer/config/sft_trainer.yaml"  # 记得修改路径

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=65536 \
     -m verl.trainer.fsdp_sft_trainer \
    --config_path=$CONFIG_PATH