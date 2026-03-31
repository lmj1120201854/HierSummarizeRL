# HierSummarizeRL 
[![W&B Report](https://img.shields.io/badge/W&B-View_Report-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://api.wandb.ai/links/lmj1120201854-beijing-institute-of-technology/uzmo4qi0)
HierSummarizeRL 是一个面向中文新闻场景的层次摘要强化学习项目，目标是让大语言模型一次性生成结构化的三级摘要，而不是只输出单一粒度的“标准摘要”。

项目基于 `verl` 训练框架实现，围绕“信息覆盖是否充分、表达是否简洁流畅、输出结构是否稳定、不同层级长度是否合适”这几个核心问题设计训练与评测流程。当前仓库同时包含数据、训练脚本、奖励函数实现、评测脚本以及模型导出脚本。

## 项目动机

传统单层摘要通常只能给出一种固定长度的结果，但真实使用场景往往需要不同摘要粒度：

- 极短摘要：用于标题补充、消息推送、卡片预览
- 短摘要：用于列表页、搜索结果页、快速浏览
- 长摘要：用于详情页导读、检索增强、下游信息消费

如果直接让模型“自由生成三级摘要”，通常会遇到几类问题：

- 三个层级的信息密度不稳定，容易出现层级之间内容重复
- 模型容易遗漏关键信息，尤其是在长新闻和多事实新闻中
- 输出格式不稳定，JSON 字段缺失或层级混乱
- 仅依赖词面匹配指标训练，难以对“覆盖了哪些关键信息点”进行细粒度约束

HierSummarizeRL 的核心思路是：先把任务形式定义清楚，再用带结构约束和奖励约束的方式，把“三级摘要生成”训练成一个可优化的目标。

## 项目特点

- 结构化三级摘要生成：统一输出 `extreme_short`、`short`、`long` 三个层级
- 面向中文新闻摘要：数据来自 NLPCC 2017 单文档摘要任务，并做了结构化加工
- 覆盖率导向奖励：不是只看 Rouge，而是显式检查摘要是否覆盖关键事实点
- 奖励可解释：覆盖率、简洁流畅、JSON 格式、长度控制分别建模
- 训练链路完整：包含 SFT、GRPO、评测和导出 Hugging Face 权重的脚本
- 基于 `verl` 扩展：在通用 RLHF/RLAIF 框架上实现了任务定制奖励

## 任务定义

模型接收新闻标题和正文，输出如下 JSON：

```json
{
  "extreme_short": "极短摘要内容",
  "short": "短摘要内容",
  "long": "长摘要内容"
}
```

当前任务约束为：

- `extreme_short`：1 句话，尽量控制在 50 字以内
- `short`：3-5 句话，尽量控制在 100 字以内
- `long`：较完整概述，尽量控制在 200 字以内

## 数据集介绍

### 原始数据

根据项目文档，原始语料来自 **NLPCC 2017 Task 3: Single Document Summarization**。每条样本包含：

- `title`：新闻标题
- `content`：新闻正文

项目文档描述的原始划分为：

- 训练集：10,000 条
- 测试集：1,000 条

此外，项目还使用 `DeepSeek-R1-0528` 为全部样本生成结构化三级摘要，作为后续训练中的“标准答案”或参考摘要。

### 仓库中已经提供的处理后数据

仓库当前包含以下几类数据文件：

| 文件 | 作用 | 当前字段 |
| --- | --- | --- |
| `data/nlpcc_data.summary_points.verifier.sft.parquet` | SFT 数据，主要用于训练覆盖率 verifier | `prompt`, `response` |
| `data/rl_data/nlpcc_data.rl.train.parquet` | GRPO 训练数据 | `title`, `content`, `summary_points`, `prompt`, `r1_response` |
| `data/rl_data/nlpcc_data.rl.test.parquet` | GRPO 验证数据 | `title`, `content`, `summary_points`, `prompt` |
| `eval/dataset/nlpcc_data_test.jsonl` | 离线评测数据 | `title`, `content`, `r1_response` |

从当前仓库文件统计来看：

- `data/nlpcc_data.summary_points.verifier.sft.parquet`：29,100 条
- `data/rl_data/nlpcc_data.rl.train.parquet`：8,686 条
- `data/rl_data/nlpcc_data.rl.test.parquet`：1,000 条
- `eval/dataset/nlpcc_data_test.jsonl`：1,000 条

这说明仓库中的 RL 训练集是经过进一步预处理或过滤后的版本，而不是简单保留原始 10,000 条。

### 数据字段的作用

`rl_data` 中几个关键字段的含义如下：

- `title` / `content`：待摘要的新闻文本
- `prompt`：三级摘要生成提示词
- `r1_response`：参考三级摘要，用于长度参考和离线评测
- `summary_points`：按层级整理好的关键信息点，用于覆盖率奖励计算

其中 `summary_points` 是这个项目非常关键的一步，它把“摘要质量”从模糊的整体判断，转成了“每个层级是否覆盖关键事实点”的可计算目标。

## 方法概览

项目整体可以理解为两个阶段：

1. 使用 SFT 训练任务相关的 verifier
2. 使用 GRPO 训练三级摘要生成模型

### 1. SFT 的作用

从当前仓库中的 SFT 数据和脚本来看，SFT 阶段的主要职责不是直接训练摘要生成器，而是训练一个 **信息覆盖判别器（coverage verifier）**。

这一点可以从以下事实上看出来：

- SFT 数据文件是 `data/nlpcc_data.summary_points.verifier.sft.parquet`
- 其中 `prompt` 不是“请生成摘要”，而是“给定摘要和关键信息点列表，判断每个信息点是否被覆盖”
- `response` 是 JSON 格式的判定结果，包含逐点分析与 `覆盖 / 部分覆盖 / 未覆盖` 结论
- 在 GRPO 脚本中，覆盖率打分服务通过 `COVER_VERIFIER_SERVER` 注入，并命名为 `HierSummarizeRL-Verifier`

因此，当前仓库里的 SFT 更适合被理解为：

- 用监督学习把一个通用大模型训练成“摘要覆盖率评估器”
- 让它学会根据 `summary_points` 对生成摘要进行结构化评估
- 为后续 GRPO 提供更稳定、可控、任务相关的奖励信号

换句话说，SFT 负责把“会看摘要”的能力训出来，GRPO 负责把“会写三级摘要”的能力推上去。

### 2. GRPO 的作用

GRPO 阶段直接优化摘要生成策略模型。当前示例脚本 `verl/scripts/run_grpo.sh` 使用：

- 生成模型：`Qwen3-8B`
- 优化算法：`GRPO`
- 奖励管理器：`custom`
- 训练数据：`data/rl_data/nlpcc_data.rl.train.parquet`

模型输入是 `title + content`，输出是包含 `extreme_short`、`short`、`long` 的三级摘要 JSON。GRPO 的目标不是简单模仿参考答案，而是在采样、比较和奖励反馈中逐步提升输出质量。

## GRPO 奖励函数设计

这是本项目最核心的部分。当前自定义奖励函数实现在：

- `verl/verl/workers/reward_manager/custom.py`
- `verl/verl/workers/reward_manager/utils/aux_rewards.py`
- `verl/verl/workers/reward_manager/utils/check_cover.py`
- `verl/verl/workers/reward_manager/utils/check_cf.py`

整体奖励由四部分组成：

### 1. 信息覆盖奖励

信息覆盖奖励衡量三级摘要是否覆盖了该层级应保留的关键事实点。

做法是：

- 使用 `summary_points` 提供每一层级的关键点列表
- 让 coverage verifier 逐点评估每个点是“覆盖 / 部分覆盖 / 未覆盖”
- 对每一层级分别计算 recall 和 precision
- 将 recall 与 precision 均匀组合成层级分数

当前实现中：

```text
ex_f1    = 0.5 * ex_recall    + 0.5 * ex_precision
short_f1 = 0.5 * short_recall + 0.5 * short_precision
long_f1  = 0.5 * long_recall  + 0.5 * long_precision

cover_reward = 0.2 * ex_f1 + 0.3 * short_f1 + 0.5 * long_f1
```

这意味着：

- 极短摘要更看重“是否抓住核心”
- 长摘要权重最高，因为它承担最完整的信息保留职责

### 2. 简洁性与流畅性奖励

这部分奖励由另一个 judge 模型给出，目标是约束：

- 是否简洁
- 是否流畅自然

当前实现会分别对三个层级给出 `Conciseness` 和 `Fluency` 分数，再做层级加权汇总。它主要解决两个问题：

- 明明覆盖了信息，但写得很啰嗦
- 结构是对的，但语言不顺、阅读体验差

### 3. JSON 格式奖励

模型输出必须是严格 JSON，并且必须恰好包含三个字段：

- `extreme_short`
- `short`
- `long`

如果 JSON 不合法，或者字段缺失、字段类型错误，就会触发格式惩罚。在当前实现中，一旦格式奖励非零，覆盖率奖励和简洁流畅奖励都会被清零，避免模型靠“内容上碰巧对了”掩盖结构错误。

### 4. 长度奖励

长度奖励分两部分：

- 隐式推理长度（CoT 长度）控制
- 三级摘要长度与参考摘要长度的对齐

当前实现中的长度总奖励为：

```text
R_length =
  0.3  * cot_length_reward(chain_len, 100, 200, 300)
  + 0.15 * length_reward(extreme_len, g_extreme_len)
  + 0.25 * length_reward(short_len, g_short_len)
  + 0.3  * length_reward(long_len, g_long_len)
```

其中：

- `cot_length_reward` 用来约束 `<think> ... </think>` 中间隐式推理的长度
- `length_reward` 用来控制各层级摘要长度不要与参考长度偏差过大
- 当某层摘要过短或过长时，会产生负奖励

这部分设计的意义在于：

- 防止模型为了追求覆盖率无节制拉长摘要
- 保持三级摘要之间的层级差异
- 让“极短 / 短 / 长”三个层级真的体现出不同的信息压缩率

### 5. 最终总奖励

当前代码中的总奖励写法为：

```text
total_reward = 0.8 * cover_reward + 0.2 * cf_reward + json_format_reward + length_reward
```

也就是说：

- 信息覆盖是主目标
- 简洁与流畅是辅助目标
- 格式正确是硬约束
- 长度控制是结构化生成的重要正则项

这种奖励设计比直接用 Rouge 做 RL 更贴合“三级摘要”任务，因为它显式建模了：

- 覆盖了什么
- 写得怎么样
- 格式是否可靠
- 层级长度是否合理

## 训练与评测流程

### SFT

当前仓库中的 SFT 入口：

- `verl/scripts/run_sft.sh`
- `verl/verl/trainer/config/sft_trainer.yaml`

它使用 `data/nlpcc_data.summary_points.verifier.sft.parquet` 作为训练数据，适合训练覆盖率 verifier。

### GRPO

当前仓库中的 GRPO 入口：

- `verl/scripts/run_grpo.sh`

它会：

- 加载 RL 训练数据
- 使用 `custom` reward manager 计算奖励
- 调用覆盖率 verifier 与简洁流畅 judge
- 对摘要生成模型执行 GRPO 更新

### 评测

评测脚本位于 `eval/` 目录：

- `eval/get_model_response.py`：批量调用模型生成三级摘要
- `eval/print_metric.py`：按 `extreme_short`、`short`、`long` 分别计算中文 ROUGE
- `eval/run.sh`：串起生成与评测

当前离线评测集为 `eval/dataset/nlpcc_data_test.jsonl`，包含 1,000 条样本和参考三级摘要。

## 仓库结构

```text
HierSummarizeRL/
├── data/
│   ├── nlpcc_data.summary_points.verifier.sft.parquet
│   └── rl_data/
│       ├── nlpcc_data.rl.train.parquet
│       └── nlpcc_data.rl.test.parquet
├── eval/
│   ├── dataset/
│   │   └── nlpcc_data_test.jsonl
│   ├── get_model_response.py
│   ├── print_metric.py
│   └── run.sh
├── to_hf/
│   ├── legacy_model_merger.py
│   └── model_merge.sh
└── verl/
    ├── scripts/
    │   ├── run_sft.sh
    │   └── run_grpo.sh
    └── verl/
        ├── trainer/
        └── workers/
```

各目录职责如下：

- `data/`：训练数据与奖励所需结构化中间数据
- `eval/`：离线推理与指标评测
- `to_hf/`：把训练输出合并导出为 Hugging Face 可加载权重
- `verl/`：训练框架主体以及本项目的自定义奖励实现

## 代码中与项目最相关的位置

如果你想快速理解这个项目，建议优先阅读以下文件：

- `verl/scripts/run_sft.sh`：SFT 训练入口
- `verl/scripts/run_grpo.sh`：GRPO 训练入口
- `verl/verl/trainer/config/sft_trainer.yaml`：SFT 训练配置
- `verl/verl/workers/reward_manager/custom.py`：总奖励聚合逻辑
- `verl/verl/workers/reward_manager/utils/aux_rewards.py`：JSON 与长度奖励
- `verl/verl/workers/reward_manager/utils/check_cover.py`：覆盖率奖励
- `verl/verl/workers/reward_manager/utils/check_cf.py`：简洁流畅奖励
- `eval/get_model_response.py`：推理提示词与输出解析
- `eval/print_metric.py`：离线 ROUGE 评测

## 总结

HierSummarizeRL 的重点不只是“让模型能写摘要”，而是让模型学会：

- 用统一 JSON 结构输出三级摘要
- 在不同粒度下保留不同层级的重要信息
- 在覆盖率、简洁性、流畅性和长度控制之间取得平衡

从仓库当前实现来看，这个项目最有辨识度的地方是：

- 把三级摘要任务结构化
- 把覆盖率评估显式化
- 把 verifier 与 GRPO 结合起来做可解释的奖励学习

如果你希望进一步扩展这个项目，最值得继续打磨的方向通常会是：

- 更强的 summary_points 自动构建
- 更稳定的 judge / verifier
- 面向中文新闻之外场景的跨领域泛化
- 更贴近业务场景的层级摘要评测指标
