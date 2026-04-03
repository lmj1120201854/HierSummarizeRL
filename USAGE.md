# 使用说明

## SFT

### 环境配置

```bash
# 环境配置
conda create -n verl-hi python=3.10 -y
conda activate verl-hi

pip install -U pip setuptools wheel

pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
pip install torch==2.7.1 torchvision==0.22.1
pip install torch_npu==2.7.1.dev20250724

pip install -r verl/requirements-npu.txt

pip install \
  "ray==2.46.0" \
  "transformers==4.52.4" \
  "numpy<2.0.0" \
  pyarrow>=15.0.0 \
  hydra-core codetiming

cd verl
pip install -e .
cd ..

# 安装vllm-npu版本
git clone --depth 1 --branch v0.11.0rc3 https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation -e .
pip install vllm-ascend==0.11.0rc0

# 两个小包 json_repair、backoff
```

> SFT是在4卡910b3服务器上运行的

### 启动命令

```bash
conda activate verl-hi
cd HierSummarizeRL/verl
bash scripts/run_sft.sh
```

---

## GRPO阶段

### 一些坑

1. 在npu上运行时，可能需要把HierSummarizeRL/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py中165行enable_chunked_prefill=False,enable_prefix_caching=False这让两个配置设置成False，因为vllm-ascend可能不支持前缀cache。

2. 运行GRPO的环境与SFT的环境并不一致，可以复用SFT的环境，但是得修改一些包。vllm==0.8.4+empty、vllm_ascend==0.8.4rc3这两个包可能不能用pip直接下载安装，需要下载源码、切换分支并用pip本地安装。配套的torch==2.5.1、torch-npu==2.5.1、torchvision==0.20.1、transformers==4.52.4，可以安装verl的requirements和requirements-npu文档安装。安装的时候先把torch和torch-npu降级到2.5.1，再重新安装vllm和vllm-ascend。在重新安装vllm和vllm-ascend的时候，pip install -e . 之前执行 rm -rf build *.egg-info。

3. npu上一定要设置actor_rollout_ref.ref.use_torch_compile=False、actor_rollout_ref.actor.use_torch_compile=False，在npu上没办法编译torch。

### 记录
训练配置：四台机器，每台机器上2张910b3，机器0使用vllm部署verifier，机器1,2,3使用ray连接和通讯，使用verl训练。
训练2个epoch需要运行110h左右，八张卡显存几乎全部跑满，推理机器0核心利用率65%-70%，但训练机器1,2,3核心利用率仅10%-20%，verl启动脚本还有待优化。