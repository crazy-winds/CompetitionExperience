# PII Data Detection

## Context

赛题地址: [PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

赛题数据: [Dataset Info](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data)

## 方法概述

### 数据集
- **source_dataset**: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data
- **nbroad**: https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated
- **moredata_dataset_fixed**: https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset/output?scriptVersionId=160542030


还实验了一些其他数据集，但是组合这三个数据集效果最好

组合方式：
- 取`source_dataset`中全是标签 "O" 的 $25\%$
- 取`source_dataset`中至少有一个非 "O" 的全部数据
- 取`nbroad`全部数据
- 取`moredata_dataset_fixed`全部数据

**生成数据集**：https://www.kaggle.com/code/minhsienweng/create-ai-generated-essays-using-llm （懒得花精力去改写，并没有在生成数据集上做一些工作）

### 建模
每次实验取组合数据集的 $20\%$ 作为验证集（单折），固定随机种子 3407，训练 20~30 个轮次

- 所有实验都使用 13 个独立标签进行训练
- 模型不定长，使用最大长度讲文本分为 N 句
  - 训练使用权重采样（两边权重高，中间权重低）的方式随机采取一句作为当前样本
  - 验证将每一句作为一个样本，根据文本的唯一 ID 过滤重复的预测
- Learning Rate 使用 5e-5 ，Cosine 使用 $\frac{1}{6}$ 的 Warmup
- 标签 "O" 的权重为 0.93
- $F_5$ 分数是按照 Batch 计算的（节省内存），最终将所有结果求均值。$F_5$ 计算方式参考(https://www.kaggle.com/code/conjuring92/pii-metric-fine-grained-eval)

**词性分析**: 使用 spacy 的词性分析为每个 token 标注词性，将整个词性分析的词性作为 Deberta 的 `type_vocab`，模型的输入加上 `token_type_idx`。词性分析对照表（https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/）

**AD-Dropout**: AD-Dropout 利用伪标签的方法来判断注意力中的高归因位置，再从高归因位置中 mask 一定数量的权重从而去达到缓解过拟合的问题。
因为 Deberta 中默认会有 $10\%$的 Attention Dropout，再加上AD-Dropout的 Dropout 使得网络训练较为困难，所以我将 Deberta 默认的 Attention Dropout 设为 0。
所以使用 `AD-Dropout` 代替 Deberta 的 `Attention Dropout`。
AD-Dropout的缺点也特别明显，因为需要前向传播两次，所以特别消耗资源。
AD-Dropout 论文代码（https://paperswithcode.com/paper/ad-drop-attribution-driven-dropout-for-robust）

### 后处理

懒死得了，没有做任何后处理

参考别人的方法：
1. 删除了所有未 "NAME_STUDENT" 标题大小写或包含数字/下划线的预测
2. 如果存在一个具有 9 位或更多位的数字的 "PHONE_NUM" 预测，请将其转换为 "ID_NUM"
3. 将 "STREET_ADDRESS" 标记之间的 "\n" 转换为 "I-STREET_ADDRESS" （由于 deberta Word Embedding删除了 "\n"，因此需要在后处理中恢复）
4. 删除 dr、mr、miss 等被标记为 "NAME_STUDENT"
5. 如果一个名称在一个文档中被多次提及，并且其中一个是 pii，请将它们全部标记为 pii
6. 如果 I- 预测前面没有 B-，则将其更改为 B-
7. 删除 Coursera、Wikipedia 和 .edu 网址作为预测


## 提交
使用软投票的方法，集成了3个模型（成绩不佳、状态不行等种种情况影响，并未过多研究加速，所以最大集成3个模型）

**加速**: ONNX-RunTime + 量化版本 （https://www.kaggle.com/code/lavrikovav/0-968-to-onnx-30-200-speedup-pii-inference）

### 无效方法
- 冻结部分 Encode 层
- 长文本训练（例如3072），被显存限制，所以未实现
- 一些 NER 的训练方式，如 CRF 等
- 不使用 BIO 标签，在提交结果时再转化为 BIO
