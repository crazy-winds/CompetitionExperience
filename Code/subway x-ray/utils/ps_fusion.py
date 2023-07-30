import numpy as np


def multiply(img1, bbox1, img2, bbox2):
    """ PS - 正片叠底
    Args:
        img1 (np.ndarray): 宽高为(H, W, C)，取值范围为(0 ~ 255)
        bbox1 (np.ndarray): gt_bbox，(N, 5) -> (x1, y1, h, w, C)
        img2 (np.ndarray): 宽高为(H, W, C)，取值范围为(0 ~ 255)
        bbox2 (np.ndarray): gt_bbox，(M, 5) -> (x1, y1, h, w, C)
    """
    img1 = img1 / 255.
    img2 = img2 / 255.
    img = img1 * img2 * 255
    bbox = np.concatenate((bbox1, bbox2), axis=0)
    
    return img, bbox


def color_burn(img1, bbox1, img2, bbox2):
    """ PS - 颜色加深
    Args:
        img1 (np.ndarray): 宽高为(H, W, C)，取值范围为(0 ~ 255)
        bbox1 (np.ndarray): gt_bbox，(N, 5) -> (x1, y1, h, w, C)
        img2 (np.ndarray): 宽高为(H, W, C)，取值范围为(0 ~ 255)
        bbox2 (np.ndarray): gt_bbox，(M, 5) -> (x1, y1, h, w, C)
    """
    img = 1 - (1 - img2 / 255) / (img1 / 255 + 0.001)

    mask_1 = img  < 0 
    mask_2 = img  > 1

    img = img * (1-mask_1)
    img = img * (1-mask_2) + mask_2
    img = img * 255
    bbox = np.concatenate((bbox1, bbox2), axis=0)
    
    return img, bbox