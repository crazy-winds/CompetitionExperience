# CompetitionExperience

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