from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluator(gt_ann_file, predict, iouType="bbox"):
    """ 计算mAP分数
    
    Args:
        gt_ann_file (str): 真实COCO标注json文件
        predict (np.ndarray): 由7维array组成
            0: Image_id
            1~4: bbox
            5: score
            6: category
        iouType (str): 计算类型，详见(https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
    """
    coco_true = COCO(gt_ann_file)
    coco_predict = coco_true.loadRes(predict)
    
    coco_evaluator = COCOeval(coco_true, coco_predict, iouType)
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    return sum(coco_evaluator.stats) / len(coco_evaluator.stats) * 100