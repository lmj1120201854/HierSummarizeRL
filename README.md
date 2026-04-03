# HierSummarizeRL 
[![W&B Report](https://img.shields.io/badge/W&B-View_Report-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://api.wandb.ai/links/lmj1120201854-beijing-institute-of-technology/uzmo4qi0)

HierSummarizeRL 是一个面向中文新闻场景的层次摘要强化学习项目，目标是让大语言模型一次性生成结构化的三级摘要，而不是只输出单一粒度的“标准摘要”。

项目基于 `verl` 训练框架实现，围绕“信息覆盖是否充分、表达是否简洁流畅、输出结构是否稳定、不同层级长度是否合适”这几个核心问题设计训练与评测流程。当前仓库同时包含数据、训练脚本、奖励函数实现、评测脚本以及模型导出脚本。

## 项目动机

传统单层摘要通常只能给出一种固定长度的结果，但真实使用场景往往需要不同摘要粒度：

- 极短摘要：用于标题补充、消息推送、卡片预览
- 短摘要：用于列表页、搜索结果页、快速浏览
- 长摘要：用于详情页导读、检索增强、下游信息消费

传统上下文学习或者SFT方法通常会遇到以下问题：
- 上下文学习：LLM绝对字数和严格格式的感知能力弱，会凑字数、丢失核心内容
- SFT：多级摘要的核心难点在于**非可导的多目标**约束。SFT 的交叉熵损失只能做局部的 Next-token 预测，无法全局把控绝对字数、JSON 格式以及宏观的语义信息覆盖率。同时，获取数万条完美的多级摘要标注数据成本极高。

而基于可验证奖励函数的强化学习会有以下优势：
- 把格式、长度这些硬约束，以及事实保真度这些软约束，全部转化为序列级别的 Reward。
- 让模型在无需完美 Ground Truth 的情况下，通过在**复合奖励函数下不断探索多目标的最优帕累托前沿**，自主学会如何在极端的压缩比下平衡‘信息密度’、‘语言流畅度’和‘格式保证’。

HierSummarizeRL 的核心思路是：
- 使用DeepSeek-R1作为教师：
  - 为全部数据生成三级摘要的ground truth。
  - 获取全部数据的关键信息点，作为奖励函数设计的基础。
  - 为SFT训练一个专用覆盖率验证器生成教师指导。
- 用DeepSeek-R1标注的SFT数据为Qwen3-8B做SFT，进行简单的知识蒸馏，把Qwen3-8B训练成一个专用的关键信息点覆盖率验证和评估的专家。
- 定义“关键信息点覆盖率奖励”、“语言简洁性和流畅性奖励”、“结构保持和长度惩罚奖励”三种混合奖励函数，使用GRPO算法增强Qwen3-8B在所定义的混合奖励函数下的偏好分布对齐。

## 项目特点

- 结构化三级摘要生成：统一输出 `extreme_short`、`short`、`long` 三个层级
- 面向中文新闻摘要：数据来自 NLPCC 2017 单文档摘要任务，并做了结构化加工
- 覆盖率导向奖励：不是只看 Rouge，而是显式检查摘要是否覆盖关键事实点
- 奖励可解释：覆盖率、简洁流畅、JSON 格式、长度控制分别建模
- 训练链路完整：包含 SFT、GRPO、评测和导出 Hugging Face 权重的脚本
- 基于 `verl` 扩展：在通用 RLHF/RLAIF/RLVR 框架上实现了任务定制奖励
- 全训练过程基于华为昇腾910计算卡，采用自部署的LLM-as-Judge和多机分布式训练

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

原始语料来自 **NLPCC 2017 Task 3: Single Document Summarization**。每条样本包含：

- `title`：新闻标题
- `content`：新闻正文

原始划分为：

- 训练集：10,000 条
- 测试集：1,000 条

### 数据集处理细节
#### 标准三级摘要答案生成
为了给RL训练提供高质量的对齐目标，项目利用 DeepSeek-R1 模型，对总计 11,000 条数据生成了结构化的三级摘要，作为“标准答案”。相关prompt如下：
````bash
你是一名专业的多级摘要生成助手。请基于提供的标题和正文内容，生成结构化的三级摘要。

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

请严格按照上述要求生成JSON格式的三级摘要，且只输出JSON，不要其他任何内容。
````

#### 关键信息点列表提取与评估基准建立
在生成标准摘要后，DeepSeek-R1 被再次调用，用于提取评估所需的关键信息点列表 (summary_points)，这一步骤是InfoCover奖励机制的基础。
- 提取目标: 将标准摘要分解为精炼、独立、不重复的信息点。
- 提取约束: 要求信息点客观、不可再分，并避免使用指示代词，以确保每个信息点都具有独立价值，从而作为量化信息覆盖度的最小评估单元。
相关prompt如下：
````bash
你是一个信息分析助手。请仔细阅读以下文档标题和摘要内容，将其分解为精炼、独立的信息点。

## 任务要求
- 将摘要内容分解为多个独立、不重复的信息点
- 每个信息点应该简洁明了，包含一个完整的事实或要素
- 避免使用指示代词（如"它"、"这个"等），尽量使用具体名称
- 消除冗余信息，确保每个信息点都有独立价值
- 信息点数量适中，以完整覆盖内容且不重复为准

## 输出格式
使用星号列表形式输出，每个信息点以*开头：
* 信息点1
* 信息点2
* 信息点3

## 示例
[标题]：科学家发现新型海洋生物

[摘要]：国际研究团队在太平洋马里亚纳海沟发现一种新型透明水母，该水母能够发光并在极端压力环境下生存，这一发现对深海生物学研究具有重要意义。

[输出]：
* 国际研究团队在太平洋马里亚纳海沟发现新型透明水母
* 新型透明水母具有发光能力
* 新型透明水母能在极端压力环境下生存
* 这一发现对深海生物学研究具有重要意义

现在请处理以下内容：
[标题]：{title}

[摘要]：{summary}

[输出]：
````

#### 检查模型训练数据集
为了训练一个模型专门检查RL过程中模型回复对关键信息点的覆盖情况，首先利用Qwen3-32B（R1生成的多级摘要比较完美，全是正样本，不符合训练要求）针对训练集生成多级摘要，然后使用DeepSeek-R1对生成的摘要进行评估，使用的prompt如下：
````bash
你将收到一个**摘要**和一个**关键信息点列表**。你的任务是逐一检查列表中的每个信息点在摘要中的覆盖情况，并判断每个信息点是**覆盖**、**部分覆盖**还是**未覆盖**。

**输入**

* [摘要]: 需要评估的文本摘要
* [关键信息点列表]: 需要验证覆盖情况的关键信息点列表

**输出**

以**严格的JSON格式**返回一个列表。对于列表中的每个信息点，输出一个包含以下字段的字典：

* `"analysis"`: 简要分析摘要如何覆盖该信息点
* `"conclusion"`: `"覆盖"`、`"部分覆盖"`或`"未覆盖"`之一，表示信息点在摘要中的覆盖状态

  * **"覆盖"**: 信息点的核心内容在摘要中明确出现且描述一致
  * **"部分覆盖"**: 摘要中提及了信息点的部分内容，但不完整或不够明确
  * **"未覆盖"**: 摘要中完全没有提及信息点的内容

---

现在，使用下面给出的**摘要**和**关键信息点列表**，分析每个信息点的覆盖情况，并严格按照要求的格式输出结果：

[摘要]: {summary}

[关键信息点列表]:
{key_points}

[输出]:
````

### 处理后的数据统计

当前包含以下几类数据文件：

| 文件 | 作用 | 当前字段 |
| --- | --- | --- |
| `data/nlpcc_data.summary_points.verifier.sft.parquet` | SFT 数据，主要用于训练覆盖率 verifier | `prompt`, `response` |
| `data/rl_data/nlpcc_data.rl.train.parquet` | GRPO 训练数据 | `title`, `content`, `summary_points`, `prompt`, `r1_response` |
| `data/rl_data/nlpcc_data.rl.test.parquet` | GRPO 验证数据 | `title`, `content`, `summary_points`, `prompt` |
| `eval/dataset/nlpcc_data_test.jsonl` | 离线评测数据 | `title`, `content`, `r1_response` |

训练数据统计：

- `data/nlpcc_data.summary_points.verifier.sft.parquet`：29,100 条
- `data/rl_data/nlpcc_data.rl.train.parquet`：8,686 条
- `data/rl_data/nlpcc_data.rl.test.parquet`：1,000 条
- `eval/dataset/nlpcc_data_test.jsonl`：1,000 条

## 方法概览

项目整体可以理解为两个阶段：

1. 使用 SFT 训练任务相关的 verifier，提升奖励评估速度和精确度
2. 使用 GRPO 训练三级摘要生成模型，对齐混合奖励函数

### 1. SFT 的作用

SFT 阶段的主要职责是训练一个 **信息覆盖判别器（coverage verifier）**。

- 用DeepSeek-R1和Qwen3-32B模型的蒸馏数据，把Qwen3-8B模型训练成“关键信息点覆盖率评估器”
- 让它学会根据 `summary_points` 对生成摘要进行结构化评估
- 为后续 GRPO 提供更快速、稳定、可控、任务相关的奖励信号

### 2. GRPO 的作用

GRPO 阶段直接优化摘要生成策略模型对齐定义的混合奖励函数。

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
- 长摘要权重最高，因为长摘要的优化难度最大。长摘要不仅要求覆盖更多关键细节，还极易出现冗余和事实幻觉。将权重倾斜给长摘要，能引导模型在复杂长文本生成中投入更多的学习“注意力”，从而提升整体的信息承载上限。

#### 覆盖率评估加速

`cover reward` 的计算瓶颈主要在 `verl/verl/workers/reward_manager/utils/check_cover.py`，评估模型需要对每个关键点打分，并且要简要分析这个关键点，但是每个问题的关键点数量差异很大，所以会导致极为严重的长尾推理效应（木桶短板效应），如果大量长文本在最后推理的话，整个系统就会等待这部分推理结果。当前实现除了保留“每条样本的 `extreme_short / short / long` 三层并行评估”之外，还对批量调度策略做了进一步优化：

- 先按回复长度从长到短排序，把长样本优先送入线程池，尽量减少尾部慢任务拖慢整体批次
- 使用 `concurrent.futures.as_completed(...)` 乱序回收结果，不再按输入顺序阻塞等待最慢样本
- 用 `result_dict` 按原始索引重组输出，保证加速后不会打乱 reward 与样本的一一对应关系
- 对单条样本的 verifier 异常做兜底处理，避免局部 API 失败拖垮整批评估

从工程效果上看，这类优化没有改变覆盖率奖励的定义方式，只是让较长文本先进行推理，短文本后推理，这样整个系统会在几乎相同时间推理完成，显著改善了 reward 阶段的吞吐。

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

它使用 `data/nlpcc_data.summary_points.verifier.sft.parquet` 作为训练数据。

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

下面给出当前几组代表性模型在测试集上的 ROUGE 结果。为了更细致地观察不同层级摘要的表现，结果分别报告了 `extreme_short`、`short`、`long` 三个层级上的 Recall、Precision 和 F1。

从结果上看：

- `Qwen3-8B-RL-step100` 在 `short` 和 `long` 层级上整体表现最好，说明 GRPO 对中短篇幅层级摘要带来了稳定增益
- `Qwen3-8B-RL-step70` 在 `extreme_short` 层级上已经显著优于原始 `Qwen3-8B`
- `Qwen3-14B` 在 `extreme_short` 的 Precision 上有竞争力，但在 `short`、`long` 层级上整体不如 RL 后的 8B 模型

#### Recall 指标

<table>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th colspan="3">Qwen3-8B</th>
      <th colspan="3">Qwen3-8B-RL-step70</th>
      <th colspan="3">Qwen3-8B-RL-step100</th>
      <th colspan="3">Qwen3-14B</th>
    </tr>
    <tr>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ROUGE-1</td>
      <td>0.641</td><td>0.562</td><td>0.579</td>
      <td>0.717</td><td>0.589</td><td>0.635</td>
      <td>0.713</td><td>0.607</td><td>0.640</td>
      <td>0.670</td><td>0.561</td><td>0.577</td>
    </tr>
    <tr>
      <td>ROUGE-2</td>
      <td>0.400</td><td>0.285</td><td>0.311</td>
      <td>0.456</td><td>0.310</td><td>0.361</td>
      <td>0.459</td><td>0.329</td><td>0.372</td>
      <td>0.412</td><td>0.281</td><td>0.308</td>
    </tr>
    <tr>
      <td>ROUGE-L</td>
      <td>0.590</td><td>0.451</td><td>0.441</td>
      <td>0.661</td><td>0.477</td><td>0.500</td>
      <td>0.658</td><td>0.497</td><td>0.504</td>
      <td>0.618</td><td>0.446</td><td>0.441</td>
    </tr>
  </tbody>
</table>

#### Precision 指标

<table>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th colspan="3">Qwen3-8B</th>
      <th colspan="3">Qwen3-8B-RL-step70</th>
      <th colspan="3">Qwen3-8B-RL-step100</th>
      <th colspan="3">Qwen3-14B</th>
    </tr>
    <tr>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ROUGE-1</td>
      <td>0.706</td><td>0.577</td><td>0.609</td>
      <td>0.687</td><td>0.588</td><td>0.618</td>
      <td>0.702</td><td>0.588</td><td>0.632</td>
      <td>0.707</td><td>0.584</td><td>0.617</td>
    </tr>
    <tr>
      <td>ROUGE-2</td>
      <td>0.443</td><td>0.301</td><td>0.336</td>
      <td>0.438</td><td>0.318</td><td>0.358</td>
      <td>0.453</td><td>0.327</td><td>0.375</td>
      <td>0.437</td><td>0.304</td><td>0.340</td>
    </tr>
    <tr>
      <td>ROUGE-L</td>
      <td>0.655</td><td>0.479</td><td>0.484</td>
      <td>0.634</td><td>0.492</td><td>0.498</td>
      <td>0.649</td><td>0.494</td><td>0.511</td>
      <td>0.657</td><td>0.484</td><td>0.492</td>
    </tr>
  </tbody>
</table>

#### F1 分数

<table>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th colspan="3">Qwen3-8B</th>
      <th colspan="3">Qwen3-8B-RL-step70</th>
      <th colspan="3">Qwen3-8B-RL-step100</th>
      <th colspan="3">Qwen3-14B</th>
    </tr>
    <tr>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
      <th>extreme-short</th>
      <th>short</th>
      <th>long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ROUGE-1</td>
      <td>0.664</td><td>0.564</td><td>0.588</td>
      <td>0.694</td><td>0.584</td><td>0.622</td>
      <td>0.700</td><td>0.593</td><td>0.632</td>
      <td>0.679</td><td>0.567</td><td>0.591</td>
    </tr>
    <tr>
      <td>ROUGE-2</td>
      <td>0.415</td><td>0.289</td><td>0.319</td>
      <td>0.442</td><td>0.311</td><td>0.356</td>
      <td>0.450</td><td>0.325</td><td>0.371</td>
      <td>0.418</td><td>0.290</td><td>0.320</td>
    </tr>
    <tr>
      <td>ROUGE-L</td>
      <td>0.612</td><td>0.459</td><td>0.457</td>
      <td>0.640</td><td>0.480</td><td>0.494</td>
      <td>0.646</td><td>0.491</td><td>0.503</td>
      <td>0.628</td><td>0.459</td><td>0.460</td>
    </tr>
  </tbody>
</table>

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
