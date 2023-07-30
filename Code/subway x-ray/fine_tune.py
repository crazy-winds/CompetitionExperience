#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

from timm.optim import Adan
from timm.scheduler import CosineLRScheduler
from timm.utils.model_ema import ModelEmaV2

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
BATCH_SIZE = 11
UPSAMPLER = 500
NUM_WORKDERS = 6
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
VALID_ANN_FILE = "data/pseudo_coco_from44_thre0.65.json"

MODEL_CFG_CONFIG = "config.py"
    
def two_stage_transform(dataloader):
    dataloader.dataset.transform = A.Compose(
        [
            A.Flip(p=0.5),
            A.RandomSizedBBoxSafeCrop(*IMAGE_SHAPE, p=1.),
            # A.PixelDropout(drop_value=114),
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

EPOCHS = 20
LR = 3e-5
WEIGHT_DECAY = 3e-5
VALID_EPOCH = 5
GRAD_ACCRUED_ITER = 1
EMA_STEP = 32

MODEL_NAME = "RTMDet"
cur_epoch = 1

copy_past_param = dict(
    num_bbox=3,
    classes=[1, 2, 4, 5, 6, 7, 8, 9],
    p=0.8
)
mosaic_param = dict(
    image_shape=IMAGE_SHAPE
)

pseudo_lable = (
    "data/test2/",
    "data/pseudo_coco_from48_thre0.6.json"
)


accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCRUED_ITER)
device = accelerator.device
# # DataLoader

# In[ ]:


train_dataloader = torch.utils.data.DataLoader(
    custom_dataset.CustomDatset(
        TRAIN_ANN_FILE,
        IMG_PREFIX,
        pseudo_lable=pseudo_lable,
        upsampler=UPSAMPLER,
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
        "data/test1/",
        # IMG_PREFIX,
    ),
    batch_size=1,
    shuffle=True,
    num_workers=1,
    collate_fn=custom_dataset.collate_fn,
)

two_stage_transform(train_dataloader)

# # Model

# In[ ]:


model = modules.Model(MODEL_CFG_CONFIG)
model.load_state_dict(torch.load("work_dir/48.pt", "cpu")["model"])
model_ema = ModelEmaV2(model, decay=0.99, device=device)


# # Runtime

# In[ ]:


optimizer = Adan(
    util.get_param_groups(model, ("norm", )),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# checkpoint = torch.load("work_dir/RTMDet_all.pt", "cpu")
# cur_epoch = checkpoint["epoch"]
# lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
# model.load_state_dict(checkpoint["model"])

# In[ ]:

model, optimizer, train_dataloader = accelerator.prepare(
    model, 
    optimizer,
    train_dataloader,
    # valid_dataloader,
)


# In[ ]:


def train_one_epoch(model, dataloader, optimizer, model_ema):
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
            optimizer.step()
            optimizer.zero_grad()
            
            if i % EMA_STEP == 0:
                model_ema.update(accelerator.unwrap_model(model))
    
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
    # Train #
    # # # # #
    loss, cls_loss, bbox_loss = train_one_epoch(model, train_dataloader, optimizer, model_ema)
    # lr_scheduler.step(epoch)
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
                # "lr_scheduler": lr_scheduler.state_dict(),
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
    # EMA #
    # # # # 
    if accelerator.is_local_main_process:
        torch.save(
            {
                "model": model_ema.module.state_dict(),
            },
            f"work_dir/{MODEL_NAME}_ema.pt"
        )