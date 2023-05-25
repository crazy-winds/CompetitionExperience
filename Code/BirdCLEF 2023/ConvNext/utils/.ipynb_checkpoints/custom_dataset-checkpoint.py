#!/usr/bin/env python
# coding: utf-8

# In[219]:


import torch
import random
import librosa
import numpy as np
import pandas as pd
import scipy


def split_data(df, valid_rate=.15, seed=3407, threshold=5):
    train_idx = []
    valid_idx = []
    for i, group in enumerate(df.groupby("primary_label")):
        idxs = group[1].index.to_list()
        if len(idxs) <= threshold:
            train_idx.append(idxs)
            continue
            
        np.random.seed(seed + i)
        np.random.shuffle(idxs)
        split_idx = int(len(idxs) * valid_rate)
        train_idx.append(idxs[split_idx:])
        valid_idx.append(idxs[:split_idx])
    
    return np.concatenate(train_idx), np.concatenate(valid_idx)


def mix(sounds):
    sounds = np.stack(sounds)
    powers = (sounds ** 2).mean(axis=1)
    power_sum = powers.sum(axis=0)
    ratios = np.sqrt(powers / power_sum)
    
    return (sounds * ratios.reshape(-1, 1)).sum(axis=0)


def pcen(E, alpha=0.98, delta=2, r=0.5, s=0.025, eps=1e-6):   
    M = scipy.signal.lfilter([s], [1, s - 1], E)
    smooth = (eps + M)**(-alpha)
    return (E * smooth + delta)**r - delta**r


class CustomDatset(torch.utils.data.Dataset):
    SR = 32_000
    NFFT = 2048
    NMEL = 128
    # NFFT = 2048
    # NMEL = 224
    FMAX = 16_000
    FMIN = 20
    HOP_LENGTH = NFFT // 4
    
    def __init__(
        self,
        df,
        cls2id,
        root="dataset/train_audio/", 
        duration=10,
        transform=None,
        audio_transform=None,
        train_mode=True
    ):
        """ 加载数据集
        
        Args:
            df (pd.DataFrame): 数据文件
            cls2id (Dict[str: int]): 类型和id的映射
            root (str): 数据的根目录
            duration (int): 音频持续时间
            transform: 图像数据处理流程
            audio_transform: 音频数据处理增强
            train_mode (bool): 是否在训练模式
        """
        self.df = df[["primary_label", "filename", "end_time"]]
        self.df = self.df[self.df["end_time"] > 2].reset_index(drop=True)
        self.cls2id = cls2id
        self.root = root
        self.duration = duration
        self.transform = transform
        self.audio_transform = audio_transform
        self.train = train_mode
        self.len = len(self.df)
        
    def __getitem__(self, idx):
        name, path, time = self.df.loc[idx]
        label = self.cls2id[name]
        time_step = self.duration
        if self.train:
            time_step = random.randint(4, self.duration)
        
        x, orig_sr = librosa.load(
            f"dataset/train_audio/{path}",
            sr=None,
            # sr=self.SR,
            offset=random.randint(0, max(time - 5, 0.)),
            duration=time_step,
        )
        if orig_sr != self.SR:
            x = librosa.resample(x, orig_sr=orig_sr, target_sr=self.SR)
        
        rand_num = random.random()
        if self.train and rand_num < .7:
            num_sample = random.randint(1, 4)
            x, label = self.mixup(x, label, num_sample)
        elif self.train and rand_num < .85:
            x, label = self.cutmix(x, label)
            
        if self.audio_transform:
            x = self.audio_transform(samples=x, sample_rate=self.SR)

        x = self.pipeline(x)
        if self.transform:
            x = self.transform(image=x)["image"]
        
        return x, label
    
    def pipeline(self, x: np.ndarray) -> np.ndarray:
        mels = librosa.feature.melspectrogram(
            y=x,
            sr=self.SR,
            n_fft=self.NFFT,
            n_mels=self.NMEL,
            fmax=self.FMAX,
            fmin=self.FMIN,
            hop_length=self.HOP_LENGTH,
        )
        
        db_map = pcen(mels).astype(np.float32)
        
        # db_map = librosa.power_to_db(mels, ref=np.max)
        # db_map = (db_map + 80) / 80
        return db_map
        
    def __len__(self):
        return self.len
    
    def mixup(self, orig_x, label, num_sample=1):
        hash_map = set()
        hash_map.add(label)
        length = orig_x.shape[0]
        dtype = orig_x.dtype
        sounds = [orig_x]
        for _ in range(num_sample):
            idx = random.randint(0, self.len - 1)
            name, path, time = self.df.loc[idx]
            label = self.cls2id[name]

            x, orig_sr = librosa.load(
                f"dataset/train_audio/{path}",
                sr=None,
                offset=random.randint(0, max(time - 5, 0.)),
                duration=5,
            )
            if orig_sr != self.SR:
                x = librosa.resample(x, orig_sr=orig_sr, target_sr=self.SR)
            
            hash_map.add(label)
            noise = np.zeros(length, dtype=dtype)
            start = random.randint(0, length - self.SR * 2)
            end = random.randint(start + self.SR * 2, length)
            shape = min(x.shape[0], end - start)
            noise[start: start + shape] = x[:(end - start)]
            sounds.append(noise)
        orig_x = mix(sounds)
        
        return orig_x, list(hash_map)
    
    def cutmix(self, orig_x, label):
        hash_map = set()
        hash_map.add(label)
        length = orig_x.shape[0]
        
        idx = random.randint(0, self.len - 1)
        name, path, time = self.df.loc[idx]
        label = self.cls2id[name]
        x, orig_sr = librosa.load(
            f"dataset/train_audio/{path}",
            sr=None,
            offset=random.randint(0, max(time - 5, 0.)),
            duration=5,
        )
        if orig_sr != self.SR:
            x = librosa.resample(x, orig_sr=orig_sr, target_sr=self.SR)
        hash_map.add(label)
        
        orig_end = random.randint(0, length - self.SR * 2)
        x = np.concatenate([orig_x[:orig_end], x[:length - orig_end]])
        
        return x, list(hash_map)
    
    
def upsample_data(df, thr=20, seed=3407):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df

