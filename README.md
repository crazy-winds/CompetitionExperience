# CompetitionExperience

**Life is not a competition, Life is about helping and inspiring others so we can each reach our potential.**

#### Kaggle - BirdCLEF 2023
2023.2 - 2023.5

成绩: 17 (银牌)

地址: <a href="https://www.kaggle.com/competitions/birdclef-2023">BirdCLEF 2023</a>

类型: 音频分类

任务: 在本次比赛中，您将使用机器学习技能通过声音识别东非鸟类。

评估指标:

![](Cache/Image/bird_.png)

方案:
- Mel图 + PCEN
- Mixup + Cutmix
- ConvNext、ConvNextV2、ImageBind
- Adan
- Cosine Learning + Warmup($\frac{1}{6}$Epoch)
- BCELoss + OHEM
- accelerate train(Multi GPU、FP16)
- SWA
  

明显提点技巧:
- 数据增强（mixup）
- SWA
- 权重平均融合

参考文献:
- [Kaggle 讨论区](https://www.kaggle.com/competitions/birdclef-2023/discussion?sort=votes)
- [PCEN](https://github.com/librosa/librosa/issues/615)
- [SWA](https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging)
____

#### 讯飞 - X光安检图像识别挑战赛2023
2023.6 - 2023.7

成绩: 14（非常不满）

地址: <a href="https://challenge.xfyun.cn/topic/info?type=Xray-2023&ch=ijcX54b">X-ray</a>

类型：目标检测

任务: 基于科大讯飞提供的真实X光安检图像集构建检测模型，对X光安检图像中的指定类别的物品进行检测，识别出物体的位置和类别。

评估指标：mAP（IoU = 0.5）

方法：
- 基本数据增强（旋转、翻转等）
- mixup + mosaic + mul_mixup
- mmdet - RTMDet_x（分辨率384或640感觉一样）
- Adan
- accelerate train(Multi GPU、FP16)

明显提点技巧:
- mixup（mul_mixup提点更多一些）
- TTA后使用WBF融合

____
#### 讯飞 - 鸟类品种识别挑战赛
2023.7 - 2023.8

成绩: 7 (勉勉强强)

地址: <a href="https://challenge.xfyun.cn/topic/info?type=bird-species&option=ssgy&ch=ijcX54b">鸟类品种识别挑战赛</a>

类型：图片分类

任务: 通过人工智能技术实现对鸟类图片的自动识别和分类。

评估指标：macro-F1 score

方法：
- 基本数据增强（旋转、翻转等）
- DINOv2（dinov2_vitl14_lc）冻结所有参数，另外训练两层MLP
- Adan
- accelerate train(Multi GPU、FP16)

明显提点技巧:
- 并无明显提点（分数已经封顶，3个人满分）
____

#### Kaggle - PII Data Detection
2024.1 - 2024.4

成绩: **_774 / 2048_** (菜)

地址: [PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

类型：NER（Named Entity Recognition）

任务: 检测给定文本中的 PII（学生姓名、作者等）。

评估指标：micro-F5 score （计算除 "O" 外标签的 F5 分数）

方法：
- Deberta-v3-base 冻结 Word Embedding
- 文本过长的按分段处理，即随机抽取文本的一段代表当前文本
- spacy词性分析代替 type_id + AD-Dropout 取代 Attention Dropout
- CrossEntropyLoss 降低 "O" 的权重
- AdamW
- accelerate train(Multi GPU、FP16)

明显提点技巧:
- 分段：使得模型可以看到整个文本，提升召回率
- 词性分析+AD-Dropout（单独使用会掉点）
- 模型集成（软投票）

参考文献:
- [AD-Dropout](https://paperswithcode.com/paper/ad-drop-attribution-driven-dropout-for-robust)
- [Baseline](https://www.kaggle.com/code/valentinwerner/915-deberta3base-training)
- [$F_\beta$ Score](https://www.kaggle.com/code/conjuring92/pii-metric-fine-grained-eval)
___

