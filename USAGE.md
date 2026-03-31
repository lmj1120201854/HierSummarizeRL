SFT:
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
当前SFT是在4卡910b3服务器上运行的
```bash
conda activate verl-hi
cd HierSummarizeRL/verl
bash scripts/run_sft.sh
```
