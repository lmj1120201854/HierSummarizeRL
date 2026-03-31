set -x

export SWANLAB_LOG_DIR=swanlog
export SWANLAB_MODE=local
# export SWANLAB_API_KEY=xxx  # 如果SWANLAB_MODE为cloud, 则写上api_key

nproc_per_node=8
CONFIG_PATH="/verl/verl/trainer/config/sft_trainer.yaml"  # 记得修改路径

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=65536 \
     -m verl.trainer.fsdp_sft_trainer \
    --config_path=$CONFIG_PATH