set -x

ulimit -n 65535

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

export WANDB_MODE=online
export WANDB_PROJECT=HierSummarizeRL
export WANDB_ENTITY=lmj1120201854-beijing-institute-of-technology
export WANDB_NAME=GRPO

# verifier配置
export COVER_VERIFIER_SERVER=127.0.0.1:8000
export COVER_VERIFIER_SERVER_NAME=HierSummarizeRL-Cover-Verifier

export CF_VERIFIER_SERVER=127.0.0.1:8001
export CF_VERIFIER_SERVER_NAME=HierSummarizeRL-CF-Verifier

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=prompt \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    data.shuffle=True \
    actor_rollout_ref.model.path=/data/home/3120245632/scow/ai/appData/mjli/Models/Qwen3_8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=custom \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.default_local_dir=/data/home/3120245632/scow/ai/appData/mjli/llm_proj/HierSummarizeRL/output/HierSummarizeRL-Qwen3-8B-RL \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=auto \
    data.train_files=/data/home/3120245632/scow/ai/appData/mjli/llm_proj/HierSummarizeRL/data/rl_data/nlpcc_data.rl.train.parquet \
    data.val_files=/data/home/3120245632/scow/ai/appData/mjli/llm_proj/HierSummarizeRL/data/rl_data/nlpcc_data.rl.test.parquet \
    trainer.total_epochs=2 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 $@
