#!/usr/bin/env python
# coding: utf-8

# In[219]:


import os
import glob
import mmcv
import torch
import random
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
    

class CustomDatset(torch.utils.data.Dataset):
    after_transform = A.Compose(
        [
            A.Normalize(),
            ToTensorV2()
        ],
    )
    
    def __init__(self, root, transform=None):
        """ 自定义加载数据集
        
        Args:
            root (str): 数据集根目录
        """
        datainfo = []
        idx2cls = sorted(os.listdir(root))
        cls2id = {name: i for i, name in enumerate(idx2cls)}
        for i, name in enumerate(idx2cls):
            datainfo += [(path, cls2id[name]) for path in glob.glob(f"{root}/{name}/*.jpg")]
        
        self.datainfo = datainfo
        self.transform = transform
        self.cls2id = cls2id
        
        # if "train" in ann_file:
        #     print(f"--- Pseudo Data Have {self.pseudo_len} ---")
        #     print(f"Use Copy: {self.copy_past_param is not None}\n"
        #           f"Use Mosaic: {self.mosaic_param is not None}\n"
        #           f"Use Double Mixup: {self.double_mixup_param is not None}\n"
        #           f"Use Mixup: {self.use_mixup}\n"
        #           f"Use Mask BG: {self.use_mask_bg}\n")
        
    def __getitem__(self, idx):
        # return self.mixup(idx)
        return self.sample_one_img(idx)
    
    def sample_one_img(self, idx):
        """ 采样单张数据增强的图片 """
        img_path, label = self.datainfo[idx]
        x = mmcv.imread(img_path, channel_order="rgb")

        if self.transform is not None:
            item = self.transform(image=x)
            x = item["image"]
        
        return x, label
        
    def __len__(self):
        return len(self.datainfo)
        
    def mixup(self, idx):
        """ 两张图片融合 """
        x1, y1 = self.sample_one_img(idx)
        sample_idx = random.randint(0, len(self) - 1)
        x2, y2 = self.sample_one_img(sample_idx)
        
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        x = (x1 * r + x2 * (1 - r))
        
        return x, (y1, y2, r)
        
    def concate(self, idx):
        """ 随机图片拼接 """
        def sample(idx):
            img = self.coco.imgs[self.images[idx]]
            x = mmcv.imread(self.img_prefix + img["file_name"], channel_order="rgb")
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img["id"]))
            bbox = []
            for ann_info in anns:
                bbox.append([*ann_info["bbox"], ann_info["category_id"]])

            if self.transform is not None:
                item = self.transform(image=x, bboxes=bbox)
                x = item["image"]
                bbox = item["bboxes"]
            
            return x, bbox, img["id"]
        
        x1, bbox1, image_id = sample(idx)
        x2, bbox2, _ = sample(random.randint(0, self.len - 1))
        
        H, W, C = x1.shape
        if random.random() > .5:
            T = A.Compose([A.Resize(H // 2, W)],bbox_params=A.BboxParams(format='coco'))
            item = T(image=x1, bboxes=bbox1)
            x1 = item["image"]
            bbox1 = item["bboxes"]
            item = T(image=x2, bboxes=bbox2)
            x2 = item["image"]
            bbox2 = item["bboxes"]
            
            x = np.concatenate((x1, x2), axis=0)
            for i in range(len(bbox2)):
                bbox2[i] = (bbox2[i][0], bbox2[i][1] + H // 2, bbox2[i][2], bbox2[i][3], bbox2[i][4])
            bbox = bbox1 + bbox2
        else:
            T = A.Compose([A.Resize(H, W // 2)],bbox_params=A.BboxParams(format='coco'))
            item = T(image=x1, bboxes=bbox1)
            x1 = item["image"]
            bbox1 = item["bboxes"]
            item = T(image=x2, bboxes=bbox2)
            x2 = item["image"]
            bbox2 = item["bboxes"]
            
            x = np.concatenate((x1, x2), axis=1)
            for i in range(len(bbox2)):
                bbox2[i] = (bbox2[i][0] + W // 2, bbox2[i][1], bbox2[i][2], bbox2[i][3], bbox2[i][4])
            bbox = bbox1 + bbox2

        return self.pipeline(x, bbox, image_id)
    
    def mosaic(self, idx):
        """ 实现数据集中的随机Mosaic
        mosaic_param: {
                "image_shape": (640, 640) # H, W
            }
        """
        idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        data = []
        for idx in idxs:
            if idx < self.len:
                d = self._sample_images(idx)
            else:
                d = self._sample_pseudo(idx - self.len)
            data.append(d)
        
        mosaic_shape = self.mosaic_param["image_shape"]
        yc, xc = [int(random.uniform(a / 3, a / 3 * 2)) for a in mosaic_shape]
        feature = np.full((*mosaic_shape, 3), self.fill_pad_value, dtype=data[0][0].dtype)
        bboxes = []
        # 顺序: 左上，右上，左下，右下
        for i, (img, box) in enumerate(data):
            h, w, c = img.shape
            if i == 0:
                scale = min(yc / h, xc / w)
                new_h, new_w = (int(h * scale), int(w * scale))
                img = A.Resize(new_h, new_w)(image=img)["image"]
                box[:, :4] = box[:, :4] * np.array([scale] * 4).reshape(1, 4)
                top, left = (yc - new_h, xc - new_w)
                feature[top: top + new_h, left: left + new_w] = img
                box[:, 0] += left
                box[:, 1] += top
            elif i == 1:
                scale = min(
                    yc / h,
                    (mosaic_shape[1] - xc) / w
                )
                new_h, new_w = (int(h * scale), int(w * scale))
                img = A.Resize(new_h, new_w)(image=img)["image"]
                box[:, :4] = box[:, :4] * np.array([scale] * 4).reshape(1, 4)
                top, right = (yc - new_h, xc + new_w)
                feature[top: top + new_h, xc: right] = img
                box[:, 0] += xc
                box[:, 1] += top
            elif i == 2:
                scale = min(
                    (mosaic_shape[0] - yc) / h,
                    xc / w
                )
                new_h, new_w = (int(h * scale), int(w * scale))
                img = A.Resize(new_h, new_w)(image=img)["image"]
                box[:, :4] = box[:, :4] * np.array([scale] * 4).reshape(1, 4)
                bottom, left = (yc + new_h, xc - new_w)
                feature[yc: bottom, left: left + new_h] = img
                box[:, 0] += left
                box[:, 1] += yc
            else:
                scale = min(
                    (mosaic_shape[0] - yc) / h,
                    (mosaic_shape[1] - xc) / w
                )
                new_h, new_w = (int(h * scale), int(w * scale))
                img = A.Resize(new_h, new_w)(image=img)["image"]
                box[:, :4] = box[:, :4] * np.array([scale] * 4).reshape(1, 4)
                bottom, right = (yc + new_h, xc + new_w)
                feature[yc: yc + new_h, xc: xc + new_w] = img
                box[:, 0] += xc
                box[:, 1] += yc
            
            bboxes.append(box)
        
        bboxes = np.concatenate(bboxes, axis=0)
        return self.pipeline(feature, bboxes)
            
    def _sample_images(self, idx):
        """ 获取图片和bbox """
        img = self.coco.imgs[self.images[idx]]
        x = mmcv.imread(self.img_prefix + img["file_name"], channel_order="rgb")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img["id"]))
        bbox = []
        for ann_info in anns:
            bbox.append([*ann_info["bbox"], ann_info["category_id"]])
        
        if self.transform is not None:
            item = self.transform(image=x, bboxes=bbox)
            x = item["image"]
            bbox = item["bboxes"]
        
        bbox = np.array(bbox).reshape(-1, 5)
        return x, bbox
            
    def _sample_pseudo(self, idx):
        """ 获取扩充的图片和bbox """
        img = self.pseudo_coco.imgs[self.pseudo_images[idx]]
        x = mmcv.imread(self.pseudo_prefix + img["file_name"], channel_order="rgb")
        anns = self.pseudo_coco.loadAnns(self.pseudo_coco.getAnnIds(imgIds=img["id"]))
        bbox = []
        for ann_info in anns:
            bbox.append([*ann_info["bbox"], ann_info["category_id"]])

        if self.transform is not None:
            item = self.transform(image=x, bboxes=bbox)
            x = item["image"]
            bbox = item["bboxes"]
        
        bbox = np.array(bbox).reshape(-1, 5)
        return x, bbox
    
    def double_mixup(self, idx):
        """ 五张图片融合
        double_mixup_param: {
                "image_shape": (640, 640) # H, W
            }
        """
        idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(4)]
        data = []
        for idx in idxs:
            if idx < self.len:
                d = self._sample_images(idx)
            else:
                d = self._sample_pseudo(idx - self.len)
            data.append(d)
        
        double_mixup_shape = self.double_mixup_param["image_shape"]
        yc, xc = (double_mixup_shape[0] // 2, double_mixup_shape[1] // 2)
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        feature = data[0][0]
        bboxes = [data[0][1]]
        # 顺序: 左上，右上，左下，右下
        for i, (img, box) in enumerate(data[1:]):
            h, w, c = img.shape
            if i == 0:
                scale = min(yc / h, xc / w)
                new_h, new_w = (int(h * scale), int(w * scale))
                img = A.Resize(new_h, new_w)(image=img)["image"]
                box[:, :4] = box[:, :4] * np.array([scale] * 4).reshape(1, 4)
                top, left = (yc - new_h, xc - new_w)
                feature[top: top + new_h, left: left + new_w] = feature[top: top + new_h, left: left + new_w] * r +  img * (1 - r)
                box[:, 0] += left
                box[:, 1] += top
            elif i == 1:
                scale = min(
                    yc / h,
                    (double_mixup_shape[1] - xc) / w
                )
                new_h, new_w = (int(h * scale), int(w * scale))
                img = A.Resize(new_h, new_w)(image=img)["image"]
                box[:, :4] = box[:, :4] * np.array([scale] * 4).reshape(1, 4)
                top, right = (yc - new_h, xc + new_w)
                feature[top: top + new_h, xc: right] = feature[top: top + new_h, xc: right] * r +  img * (1 - r)
                box[:, 0] += xc
                box[:, 1] += top
            elif i == 2:
                scale = min(
                    (double_mixup_shape[0] - yc) / h,
                    xc / w
                )
                new_h, new_w = (int(h * scale), int(w * scale))
                img = A.Resize(new_h, new_w)(image=img)["image"]
                box[:, :4] = box[:, :4] * np.array([scale] * 4).reshape(1, 4)
                bottom, left = (yc + new_h, xc - new_w)
                feature[yc: bottom, left: left + new_h] = feature[yc: bottom, left: left + new_h] * r +  img * (1 - r)
                box[:, 0] += left
                box[:, 1] += yc
            else:
                scale = min(
                    (double_mixup_shape[0] - yc) / h,
                    (double_mixup_shape[1] - xc) / w
                )
                new_h, new_w = (int(h * scale), int(w * scale))
                img = A.Resize(new_h, new_w)(image=img)["image"]
                box[:, :4] = box[:, :4] * np.array([scale] * 4).reshape(1, 4)
                bottom, right = (yc + new_h, xc + new_w)
                feature[yc: yc + new_h, xc: xc + new_w] = feature[yc: yc + new_h, xc: xc + new_w] * r +  img * (1 - r)
                box[:, 0] += xc
                box[:, 1] += yc
            
            bboxes.append(box)
        
        bboxes = np.concatenate(bboxes, axis=0)
        return self.pipeline(feature, bboxes)
    
    def mask_bg(self, image, bboxes):
        """ 随机将非gt_bbox区域mask掉
        
        Args:
            image (np.ndarray)：图片(H, W, C)
            bboxes (np.ndarray): gt_bbox(N, 5)，标签为np.ndarray(xywh, cls)
        """
        H, W, C = image.shape
        new_img = np.full(image.shape, self.fill_pad_value, dtype=image.dtype)
        
        for i in range(len(bboxes)):
            x1, y1, w, h, *_ = bboxes[i]
            x2, y2 = (x1 + w, y1 + h)
            bx1, by1 = x1 * random.random(), y1 * random.random()
            bx2, by2 = (x2 + (W - x2) * random.random(), y2 + (H - y2) * random.random())
            bx2, by2 = (np.clip(bx2, 0, W), np.clip(by2, 0, H))
            bx1, by1, bx2, by2 = (
                bx1.astype(np.int32),
                by1.astype(np.int32),
                bx2.astype(np.int32),
                by2.astype(np.int32),
            )
            new_img[by1: by2, bx1: bx2] = image[by1: by2, bx1: bx2]
        
        return new_img, bboxes
    
    def mul_mixup(self, idx):
        """ 使用乘法的mixup """
        idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(1)]
        data = []
        for idx in idxs:
            if idx < self.len:
                d = self._sample_images(idx)
            else:
                d = self._sample_pseudo(idx - self.len)
            data.append(d)
            
        inputs, bboxes = multiply(*data[0], *data[1])
        return self.pipeline(inputs, bboxes)