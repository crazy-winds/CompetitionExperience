#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

from timm.optim import Adan
from timm.scheduler import CosineLRScheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import tqdm
import numpy as np
from accelerate import Accelerator

from utils import (
    util,
    modules,
    coco_metrics,
    combo_augment,
    custom_dataset, 
)

torch.set_float32_matmul_precision('high')


# # Parameters

# In[ ]:
BATCH_SIZE = 13
UPSAMPLER = 500
NUM_WORKDERS = 10
# IMAGE_SHAPE = (768, 768)
IMAGE_SHAPE = (384, 384)
# IMAGE_SHAPE = (640, 640)
# IMG_PREFIX = "data/train/"
# TRAIN_ANN_FILE = "data/train/coco_ann.json"
# VALID_ANN_FILE = "data/train/coco_ann.json"
# TRAIN_ANN_FILE = "data/train_coco.json"
# VALID_ANN_FILE = "data/valid_coco.json"
IMG_PREFIX = "sift_data/images/"
TRAIN_ANN_FILE = "sift_data/coco_ann.json"
VALID_ANN_FILE = "sift_data/valid_coco.json"

def one_stage_transform(dataloader):
    dataloader.dataset.transform = A.Compose(
        [
            A.OneOf(
                [
                    A.SmallestMaxSize(max_size=400, p=.25),
                    A.SmallestMaxSize(max_size=480, p=.25),
                    A.SmallestMaxSize(max_size=576, p=.25),
                    A.SmallestMaxSize(max_size=768, p=.25),
                ],
                p=.6
            ),
            A.RandomSizedBBoxSafeCrop(*IMAGE_SHAPE, p=1.),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=180, scale_limit=.2, value=255, p=.6),
            A.PixelDropout(drop_value=255),
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=255, p=0.5),
            
            A.OneOf([
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.25),
        ],
        bbox_params=A.BboxParams(
            format='coco',
        )
    )
    
def two_stage_transform(dataloader):
    dataloader.dataset.transform = A.Compose(
        [
            A.Flip(p=0.5),
            A.RandomSizedBBoxSafeCrop(*IMAGE_SHAPE, p=1.),
            # A.PixelDropout(drop_value=255),
        ],
        bbox_params=A.BboxParams(
            format='coco',
        )
    )
    dataloader.dataset.use_mixup = False
    dataloader.dataset.use_mask_bg = False
    dataloader.dataset.mosaic_param = None
    dataloader.dataset.copy_past_param = None
    dataloader.dataset.double_mixup_param = None

    
valid_transform = A.Compose(
    [
        # A.SmallestMaxSize(max_size=VALID_RESOLUTION, p=1.),
        # A.CenterCrop(VALID_RESOLUTION, VALID_RESOLUTION, p=1.),
    ],
    bbox_params=A.BboxParams(
        format='coco',
    )
)

EPOCHS = 60
LR = 1e-4
WARMUP_LR_INIT = 1e-5
WEIGHT_DECAY = 3e-5
VALID_EPOCH = 5
GRAD_ACCRUED_ITER = 1
swa_start = EPOCHS // 5 * 4
swa_freq = 5
STAGE_EPOCH = EPOCHS - 20

MODEL_NAME = "RTMDet"
MODEL_CFG_CONFIG = "config.py"
cur_epoch = 1

# # # # # # # # #
# 强数据增强策略 #
# # # # # # # # 
copy_past_param = dict(
    classes=[1, 2, 4, 5, 6, 7, 8, 9],
    add_box_num=4,
    p=.8
)

mosaic_param = dict(
    image_shape=IMAGE_SHAPE
)

pseudo_label = (
    "data/test2/",
    "data/pseudo_coco_from814_thre0.6.json"
)

# copy_past_param = None
# mosaic_param = None
# pseudo_label = None


accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCRUED_ITER)
device = accelerator.device
# # DataLoader

# In[ ]:


train_dataloader = torch.utils.data.DataLoader(
    custom_dataset.CustomDatset(
        TRAIN_ANN_FILE,
        IMG_PREFIX,
        # copy_past_param=copy_past_param,
        mosaic_param=mosaic_param,
        double_mixup_param=mosaic_param,
        pseudo_lable=pseudo_label,
        upsampler=UPSAMPLER,
        use_mixup=True,
        use_mask_bg=True,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKDERS,
    collate_fn=custom_dataset.collate_fn,
    pin_memory=True
)

valid_dataloader = torch.utils.data.DataLoader(
    custom_dataset.CustomDatset( 
        VALID_ANN_FILE,
        IMG_PREFIX,
    ),
    batch_size=1,
    shuffle=True,
    num_workers=1,
    collate_fn=custom_dataset.collate_fn,
)


# # Model

# In[ ]:


model = modules.Model(MODEL_CFG_CONFIG)
model.load_state_dict(torch.load("work_dir/RTMDet_model_model.pt", "cpu")["model"])


# # Runtime

# In[ ]:


optimizer = Adan(
    util.get_param_groups(model, ("norm", )),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)
lr_scheduler = CosineLRScheduler(
    optimizer,
    EPOCHS,
    warmup_t=EPOCHS // 6,
    warmup_lr_init=WARMUP_LR_INIT,
)

# checkpoint = torch.load("work_dir/RTMDet_all.pt", "cpu")
# cur_epoch = checkpoint["epoch"]
# lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
# model.load_state_dict(checkpoint["model"])

# In[ ]:
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, 
    optimizer,
    train_dataloader,
    # valid_dataloader,
    lr_scheduler,
)
# awp = util.AWP(model, optimizer, adv_lr=.1, platform=accelerator)


# In[ ]:


def train_one_epoch(model, dataloader, optimizer, use_awp=False):
    dataloader = tqdm.tqdm(dataloader, disable=not accelerator.is_main_process)
    model.train()

    losses = 0
    cls_losses = 0
    bbox_losses = 0
    for i, batch in enumerate(dataloader, 1):
        with accelerator.accumulate(model):
            cls_loss, bbox_loss = model(batch, mode="loss")
            cls_losses += cls_loss.item()
            bbox_losses += bbox_loss.item()
            losses += (cls_loss.item() + bbox_loss.item())
            accelerator.backward(cls_loss + bbox_loss)
            # if use_awp:
            #     awp.attack_backward(batch)
            optimizer.step()
            optimizer.zero_grad()
    
    return (
        losses / len(dataloader),
        cls_losses / len(dataloader),
        bbox_losses / len(dataloader),
    )


# In[ ]:


@torch.no_grad()
def valid_one_epoch(model, dataloader):
    dataloader = tqdm.tqdm(dataloader, disable=not accelerator.is_main_process)
    
    model.eval()
    predict = []
    for batch in dataloader:
        out = model(batch, mode="predict")
        for data_sample in out:
            image_id = data_sample.image_id
            pred_instances = data_sample.pred_instances
            for i in range(len(pred_instances.labels)):
                bbox = pred_instances.bboxes[i].detach().cpu()
                bbox[2:] = bbox[2:] - bbox[:2]
                predict.append(
                    np.array([
                        image_id,
                        *bbox,
                        pred_instances.scores[i].detach().cpu(),
                        pred_instances.labels[i].detach().cpu()
                    ])
                )

    if len(predict) == 0:
        print("----" * 50)
        print("未匹配到检测框")
        return 0
    
    output = np.stack(predict, axis=0)
    return coco_metrics.evaluator(VALID_ANN_FILE, output)


# In[ ]:

min_loss = float("inf")
max_score = 0
model.train()
for epoch in range(cur_epoch, EPOCHS + 1):
    # # # # #
    # Stage #
    # # # # #
    if STAGE_EPOCH >= cur_epoch:
        one_stage_transform(train_dataloader)
    else:
        two_stage_transform(train_dataloader)
        
    # # # # #
    # Train #
    # # # # #
    loss, cls_loss, bbox_loss = train_one_epoch(model, train_dataloader, optimizer, max_score > 70)
    # loss, cls_loss, bbox_loss = train_one_epoch(model, train_dataloader, optimizer, True)
    lr_scheduler.step(epoch)
    accelerator.print(
        f"Epoch {epoch} - Loss: {loss :.4f}"
        f"\tlr: {optimizer.param_groups[0]['lr']*1e4 :.4f}\t"
        f"cls_loss: {cls_loss :.3f}\t"
        f"bbox_loss: {bbox_loss :.3f}"
    )
    if min_loss > loss:
        min_loss = loss
    # if epoch == int(EPOCHS // 6):
    #     util.update_dropout_rate(model, .0)
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        torch.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            },
            f"work_dir/{MODEL_NAME}_all.pt" 
        )

        torch.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                # "ema": ema_model.module.state_dict()
            },
            f"work_dir/{MODEL_NAME}_model.pt"
        )

    # # # # #
    # Valid #
    # # # # #
    if epoch % VALID_EPOCH == 0:
        s = valid_one_epoch(model, valid_dataloader)
        if s > max_score:
            torch.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    # "ema": ema_model.module.state_dict()
                },
                f"work_dir/{MODEL_NAME}_best.pt"
            )
            max_score = s
        print(f"Valid Epoch {epoch} - Score: {s :.2f}%\tMax Score: {max_score :.2f}%")
        
    # # # #
    # SWA #
    # # # #
    # if accelerator.is_local_main_process and swa_start == epoch:
    #     swa_model = torch.optim.swa_utils.AveragedModel(accelerator.unwrap_model(model))
    # if accelerator.is_local_main_process and swa_start <= epoch and epoch % swa_freq == 0:
    #     swa_model.update_parameters(model)
    #     torch.save(
    #         {
    #             "model": swa_model.module.state_dict()
    #         },
    #         f"work_dir/{MODEL_NAME}_swa.pt"
    #     )


# if accelerator.is_local_main_process:
#     valid_one_epoch(model, valid_dataloader)