#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import tqdm
import numpy as np
import pandas as pd
from timm.optim.adan import Adan
from timm.utils.model_ema import ModelEmaV2
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss.asymmetric_loss import AsymmetricLossMultiLabel

from accelerate import Accelerator
import albumentations as A
import audiomentations as Audio
from albumentations.pytorch import ToTensorV2

from utils import (
    util,
    losses,
    modules,
    custom_dataset,
)

import gc
import glob
import sklearn.metrics

# 忽略外部的采样率为48k, 而数据集采样率32k
import warnings
warnings.filterwarnings("ignore")



# In[ ]:


torch.set_float32_matmul_precision('high')


# In[ ]:


accelerator = Accelerator()
device = accelerator.device


# # Parameters

# In[ ]:


MODEL_NAME = "ConvNextS"

id2cls = os.listdir("dataset/train_audio")
id2cls.sort()
cls2id = {v: k for k, v in enumerate(id2cls)}

LR = 1e-4
LR_MIN = 1e-5
EPOCHS = 70
VALID_EPOCH = 10
BATCH_SIZE = 42
NUM_WORKER = 6
WEIGHT_DECAY = 3e-2
DROPOUT = .1

valid_rate = .1
test_duration = 5
train_duration = 10
# import random
# split_data_seed = random.randint(1, 33333)
split_data_seed = 3407

MIN_SAMPLE = 50

cur_epoch = 0
swa_start = 0
swa_freq = 10


# In[ ]:
audio_transform = Audio.Compose(
    [
        # Audio.AirAbsorption(
        #     min_distance=5,
        #     max_distance=300,
        #     p=.5
        # ),
        Audio.OneOf(
            [
                Audio.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                # Audio.AddBackgroundNoise(
                #     sounds_path=glob.glob("outside_dataset/audio/*.mp3"),
                #     min_snr_in_db=3.0,
                #     max_snr_in_db=40.0,
                #     noise_transform=Audio.PolarityInversion(),
                #     p=0.5
                # ),
            ],
            p=.5
        ),
        Audio.OneOf(
            [
                Audio.Gain(),
                Audio.GainTransition()
            ],
            p=.4
        ),
        # 右边拉伸缩短
        Audio.TimeStretch(min_rate=0.8, max_rate=1.25, p=.5),
        # 上下移动
        # Audio.PitchShift(min_semitones=-6, max_semitones=6, p=.5),
        # 滚轮滑动
        Audio.Shift(min_fraction=-0.5, max_fraction=0.5, p=.5),
        # 时间轴 mask
        Audio.TimeMask(min_band_part=0.1, max_band_part=0.3, fade=True, p=.5),
        
        Audio.OneOf(
            [
                # 分贝普遍降低
                Audio.LowPassFilter(50, 5000, p=.5),
                # 分贝普遍提高
                Audio.HighPassFilter(50, 2000, p=.5),
                # 分贝随机高低
                Audio.BandPassFilter(50, 2000, p=.5),
                Audio.BandStopFilter(p=.5)
            ],
            p=.4
        )
        
    ]
)

spec_transofrm = Audio.SpecFrequencyMask(p=.5)

train_transform = A.Compose([
    A.Resize(128, 204),
    # A.Resize(128, 312),
    # A.Resize(126, 308),
    ToTensorV2(),
])
test_transform = A.Compose([
    A.Resize(128, 204),
    # A.Resize(128, 312),
    # A.Resize(126, 308),
    ToTensorV2(),
])


# # Dataset

# In[ ]:


data = pd.read_csv("dataset/time_audio.csv")
train_idx, valid_idx = custom_dataset.split_data(data, valid_rate, split_data_seed)
# train_data = data.loc[train_idx].reset_index(drop=True)
train_data = data
train_data = custom_dataset.upsample_data(train_data, MIN_SAMPLE)
valid_data = data.loc[valid_idx].reset_index(drop=True)
valid_data = pd.concat(
    [valid_data, valid_data, valid_data, valid_data,
     valid_data, valid_data, valid_data, valid_data],
    axis=0,
    ignore_index=True
)


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch])
    y = [b[1] for b in batch]
    
    return x, y



train_dataloader = torch.utils.data.DataLoader(
    custom_dataset.CustomDatset(
        train_data,
        cls2id=cls2id,
        root="dataset/train_audio/",
        duration=train_duration,
        transform=train_transform,
        audio_transform=audio_transform,
        spec_transofrm=spec_transofrm,
        train_mode=True,
    ),
    collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    shuffle=True,
    num_workers=NUM_WORKER,
)

valid_dataloader = torch.utils.data.DataLoader(
    custom_dataset.CustomDatset(
        valid_data,
        cls2id=cls2id,
        root="dataset/train_audio/",
        duration=test_duration,
        transform=test_transform,
        train_mode=False
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKER,
)

del data, train_data, valid_data, train_idx, valid_idx
gc.collect()


# # Model

# In[ ]:
0

# model = modules.ConvNext(len(id2cls), pretrained=True, dropout=DROPOUT)
# model.load_state_dict(torch.load("checkpoint/ConvNextS_best.pt", "cpu")["model"])
# model = modules.Dinov2(len(id2cls), pretrained=True, dropout=DROPOUT)
# model = modules.ImageBindAudio(len(id2cls), "ImageBind/checkpoints/imagebind_huge.pth")
model = modules.ImageBindAudio(len(id2cls), None)
model.load_state_dict("checkpoint/ConvNextS_best.pt")

# # Misc

# In[ ]:

    
class Criterion(nn.BCEWithLogitsLoss):
    def __init__(self, sample_num=10, **kwargs):
        super().__init__(**kwargs)
        self.reduction = "none"
        self.sample_num = sample_num
        
    def forward(self, x, y):
        b = x.shape[0]
        m = torch.zeros_like(x, device=device) + 1e-4
        for i in range(b):
            m[i, y[i]] = .98
        loss = super().forward(x, m)
        ohem_loss = loss.sort(-1, True).values[:, :self.sample_num]
        
        return loss.sum(dim=1).mean() + ohem_loss.sum(dim=1).mean() * .7

    
# class Criterion(nn.BCEWithLogitsLoss):
#     def __init__(self, sample_num=5, normalize_targets=True, **kwargs):
#         super().__init__(**kwargs)
#         self.reduction = "none"
#         self.sample_num = sample_num
        
#         self._normalize_targets = normalize_targets
#         self._eps = torch.finfo(torch.float32).eps
        
#     def forward(self, x, y):
#         b = x.shape[0]
#         m = torch.zeros_like(x, device=device)
#         for i in range(b):
#             m[i, y[i]] = 1.

#         if self._normalize_targets:
#             m /= self._eps + m.sum(dim=1, keepdim=True)
#         per_sample_per_target_loss = -m * F.log_softmax(x, -1)
#         per_sample_loss = torch.sum(per_sample_per_target_loss, -1).sum()

#         return per_sample_loss


# In[ ]:

criterion = Criterion()
# optimizer = Adan(
optimizer = optim.AdamW(
    util.get_param_groups(model, ("norm",)),
    # util.get_param_groups(model),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
# lr_scheduler = CosineLRScheduler(
#     optimizer,
#     EPOCHS,
#     warmup_lr_init=LR_MIN,
#     warmup_t=EPOCHS // 8,
# )
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    EPOCHS,
    2
)

# In[ ]:


model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
    model, 
    optimizer,
    train_dataloader,
    valid_dataloader,
    lr_scheduler
)


# # Train

# In[ ]:

def save_model(model, model_name):
    model = accelerator.unwrap_model(model)
    # torch.save(
    #     {
    #         "model": model.state_dict(),
    #     },
    #     model_name
    # )
    torch.save(
        {
            "preprocessors": model.preprocessors.state_dict(),
            "trunk": model.trunk.state_dict(),
            "head": model.head.state_dict(),
            "postprocessors": model.postprocessors.state_dict(),
            "fc": model.fc.state_dict(),
        },
        model_name
    )


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    # dataloader = tqdm.tqdm(dataloader, disable=not accelerator.is_main_process)
    
    losses = 0
    for i, (x, y) in enumerate(dataloader, 1):
        optimizer.zero_grad(set_to_none=True)
            
        output = model(x)
        loss = criterion(output, y)
        losses += loss.item()
        
        accelerator.backward(loss)
        optimizer.step()
        
    
    return losses / len(dataloader)


# In[ ]:


@torch.no_grad()
def valid_one_epoch(model, dataloader):
    def padded_cmap(submission, solution, padding_factor=5):
        b, c = solution.shape
        new_rows = np.ones((padding_factor, c))
        padded_solution = np.concatenate((solution, new_rows), axis=0)
        padded_submission = np.concatenate((submission, new_rows), axis=0)
        score = sklearn.metrics.average_precision_score(
            padded_solution,
            padded_submission,
            average='macro',
        )
        
        return score
    
    model.eval()
    # dataloader = tqdm.tqdm(dataloader, disable=not accelerator.is_main_process)
    
    output = []
    target = []
    for x, y in dataloader:
        if y.dim() == 1:
            y = y.reshape(-1, 1)
        out = model(x).cpu()
        output.append(out.sigmoid())
        target.append(torch.zeros_like(out).scatter(1, y.cpu(), 1.))
        
    score = padded_cmap(
        torch.cat(output).numpy(),
        torch.cat(target).numpy(),
        1
    )
    
    return score * 100


# In[ ]:


min_loss = float("inf")
max_score = 0
for epoch in range(cur_epoch, EPOCHS + 1):
    # # # # #
    # Train #
    # # # # #
    loss = train_one_epoch(model, train_dataloader, optimizer, criterion)
    accelerator.wait_for_everyone()
    lr_scheduler.step(epoch)
    # accelerator.print(f"Epoch {epoch} - Loss: {loss :.4f}\tlr: {optimizer.param_groups[0]['lr']*1e4 :.4f}")
    if not accelerator.is_main_process:
        with open("log.txt", "a") as f:
            f.write(f"Epoch {epoch} - Loss: {loss :.4f}\tlr: {optimizer.param_groups[0]['lr']*1e4 :.4f}\n")
    if min_loss > loss:
        min_loss = loss
    # if epoch == int(EPOCHS // 6):
    #     util.update_dropout_rate(model, .0)
    if accelerator.is_local_main_process:
        # torch.save(
        #     {
        #         "model": accelerator.unwrap_model(model).state_dict(),
        #         "lr_scheduler": lr_scheduler.state_dict(),
        #         "epoch": epoch,
        #     },
        #     f"checkpoint/{MODEL_NAME}_all.pt"
        # )

        save_model(model, f"checkpoint/{MODEL_NAME}_model.pt")
    
    # # # # #
    # Valid #
    # # # # #
    if epoch % VALID_EPOCH == 0:
        s = valid_one_epoch(model, valid_dataloader)
        if s > max_score and accelerator.is_local_main_process:
            save_model(model, f"checkpoint/{MODEL_NAME}_best.pt")
        max_score = max(s, max_score)
        # accelerator.print(f"Valid Epoch {epoch} - Score: {s :.2f}%\tMax Score: {max_score :.2f}%")
        if not accelerator.is_main_process:
            with open("log.txt", "a") as f:
                f.write(f"Valid Epoch {epoch} - Score: {s :.2f}%\tMax Score: {max_score :.2f}%\n")
        
    # # # #
    # SWA #
    # # # #
    if swa_start == epoch:
        swa_model = torch.optim.swa_utils.AveragedModel(accelerator.unwrap_model(model))
    if swa_start <= epoch and epoch % swa_freq == 0:
        swa_model.update_parameters(model)
        save_model(swa_model.module, f"checkpoint/{MODEL_NAME}_swa.pt")
