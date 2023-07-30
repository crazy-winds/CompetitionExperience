import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def vision_batch0(batch):
    """ 可视化batch的第一张图片 """
    x = batch["inputs"][0].numpy()
    y = batch["data_samples"][0]
    x = cv.UMat(x.transpose(1, 2, 0))
    bboxes = y.gt_instances.bboxes.long().numpy()
    print(y.gt_instances.labels)
    for i in range(len(bboxes)):
        cv.rectangle(x, bboxes[i, :2], bboxes[i, 2:], (0, 0, 255), 5)
        
    plt.imshow(cv.UMat.get(x))
    
    
def vision_output0(batch, output, threshold=.4):
    """ 可视化模型输出的第一张图片
    
    Args:
        batch: Dataloader迭代的batch
        output: 模型预测输出
        threshold (float): 低于这个阈值的框不显示 
    """
    x = batch["inputs"][0].numpy()
    mean = np.array((0.485, 0.456, 0.406)).reshape(-1, 1, 1)
    std = np.array((0.229, 0.224, 0.225)).reshape(-1, 1, 1)
    x = x * std + mean
    y = output[0]

    canvas = cv.UMat(x.copy().transpose(1, 2, 0))
    # gt
    bboxes = y.gt_instances.bboxes.long().numpy()
    for i in range(len(bboxes)):
        cv.rectangle(canvas, bboxes[i, :2], bboxes[i, 2:], (0, 0, 255), 2)
        cv.putText(canvas, str(y.gt_instances.labels[i].long().item()), bboxes[i, :2] - 1, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))

    # predict
    predict = y.pred_instances
    for i in range(len(predict.scores)):
        if predict.scores[i] < threshold:
            print(i)
            break
        bbox = predict.bboxes[i].long().numpy()
        cv.rectangle(canvas, bbox[:2], bbox[2:], (255, 0, 0), 2)
        cv.putText(canvas, str(predict.labels[i].item()), bbox[:2] - 2, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
    
    plt.imshow(cv.UMat.get(canvas))
    
    
def vision_infer(batch, output, threshold=.4):
    """ 可视化推理结果
    
    Args:
        batch: Dataloader
        output: 模型预测输出
        threshold (float): 低于这个阈值的框不显示 
    """
    x = batch["inputs"][0].numpy()
    mean = np.array((0.485, 0.456, 0.406)).reshape(-1, 1, 1)
    std = np.array((0.229, 0.224, 0.225)).reshape(-1, 1, 1)
    x = x * std + mean
    y = output[0]

    canvas = cv.UMat(x.copy().transpose(1, 2, 0))

    # predict
    predict = y.pred_instances
    for i in range(len(predict.scores)):
        if predict.scores[i] < threshold:
            print(i)
            break
        bbox = predict.bboxes[i].long().numpy()
        cv.rectangle(canvas, bbox[:2], bbox[2:], (255, 0, 0), 2)
        cv.putText(canvas, str(predict.labels[i].item()), bbox[:2] - 2, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
    
    plt.imshow(cv.UMat.get(canvas))
    