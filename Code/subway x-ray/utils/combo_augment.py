#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 参考 `MedAugment`，代码见 `https://github.com/NUS-Tim/MedAugment_Pytorch/blob/master/utils/medaugment.py`


# In[1]:


import math
import random

import albumentations as A


# In[2]:


def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num


# In[3]:


# 数据增强的强弱程度    弱 1<--->5 强
level = 3.

# 像素变换（不影响 gt_box）
PIXEL_AUGMETN = A.Compose([
    A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
    A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
    A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
    A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
    A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
    A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
    
    
    A.PixelDropout(drop_value=114),
    A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=114, p=0.2 * level),
])


# 空间变换（影响 gt_box）
SPACE_AUGMENT = A.Compose([
    A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                 crop_border=False, p=0.2 * level),
    A.HorizontalFlip(p=0.2 * level),
    A.VerticalFlip(p=0.2 * level),
    A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
             shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=114, mode=0, fit_output=False,
             keep_ratio=True, p=0.2 * level),
    A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
             shear={'x': (0, 2 * level), 'y': (0, 0)}
             , interpolation=1, mask_interpolation=0, cval=0, cval_mask=114, mode=0, fit_output=False,
             keep_ratio=True, p=0.2 * level),  # x
    A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
             shear={'x': (0, 0), 'y': (0, 2 * level)}
             , interpolation=1, mask_interpolation=0, cval=0, cval_mask=114, mode=0, fit_output=False,
             keep_ratio=True, p=0.2 * level),
    A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
             shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=114, mode=0, fit_output=False,
             keep_ratio=True, p=0.2 * level),
    A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
             shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=114, mode=0, fit_output=False,
             keep_ratio=True, p=0.2 * level),
    
    A.ShiftScaleRotate(rotate_limit=36 * level, scale_limit=.08 * level, value=114, p=0.2 * level),
    A.Flip(p=0.2 * level),
])


# In[23]:


def combo_policy(policy=[(1, 2), (0, 3), (0, 2), (1, 1)], p=.5):
    """ 随机组合数据增强策略
    
    Args:
        policy (List[Tuple[int]]): 组合的策略，第一个数字随机采样
            像素变换增强，第二个数字机采样空间变换增强
        p (float): 应用这个数据增强的概率
            
    Return:
        A.Compose(...): 随机组合的数据增强
    """
    random.shuffle(policy)
    transformers = []
    for num1, num2 in policy:
        pixel, space = (
            random.sample(PIXEL_AUGMETN.transforms, num1),
            random.sample(SPACE_AUGMENT.transforms, num2)
        )
        tf = A.Compose(
            [
                *pixel,
                *space
            ]
        )
        random.shuffle(tf.transforms)
        transformers.append(tf)
    
    return A.OneOf(transformers, p=p)

