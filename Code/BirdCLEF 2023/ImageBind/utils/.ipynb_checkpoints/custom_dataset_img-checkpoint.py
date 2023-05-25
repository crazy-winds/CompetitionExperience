#!/usr/bin/env python
# coding: utf-8


import torch
import random
import librosa
import numpy as np
import pandas as pd
import scipy

import torchaudio
from torchvision import transforms

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10

def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    # if abs(p) / n_frames > 0.2:
    #     logging.warning(
    #         "Large gap between audio n_frames(%d) and "
    #         "target_length (%d). Is the audio_target_length "
    #         "setting correct?",
    #         n_frames,
    #         target_length,
    #     )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def load_and_transform_audio_data(
    auido_array,
    num_mel_bins=128,
    target_length=304,
    sample_rate=32000,
    mean=-4.268,
    std=9.138,
):
    waveform_melspec = waveform2melspec(
        torch.from_numpy(auido_array).reshape(1, -1), sample_rate, num_mel_bins, target_length
    )

    return waveform_melspec
    # normalize = transforms.Normalize(mean=mean, std=std)(waveform_melspec)
    # return normalize


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
        spec_transofrm=None,
        train_mode=True,
        num_sample=None,
        seed=3407
    ):
        """ 加载数据集
        
        Args:
            df (pd.DataFrame): 数据文件
            cls2id (Dict[str: int]): 类型和id的映射
            root (str): 数据的根目录
            duration (int): 音频持续时间
            transform: 图像数据处理流程
            audio_transform: 音频数据处理增强
            spec_transofrm: mel图数据增强
            train_mode (bool): 是否在训练模式
        """
        self.df = df[["primary_label", "filename", "end_time"]]
        self.df = self.df[self.df["end_time"] > 1].reset_index(drop=True)
        if num_sample:
            self.df = upsample_data(self.df, num_sample, seed)
        self.cls2id = cls2id
        self.root = root
        self.duration = duration
        self.transform = transform
        self.audio_transform = audio_transform
        self.spec_transofrm = spec_transofrm
        self.train = train_mode
        self.len = len(self.df)
        self.tensor_transform = transforms.Compose([
            transforms.Resize((128, 204)),
            transforms.Normalize(mean=-4.268, std=9.138)
        ])
        
    def __getitem__(self, idx):
        name, path, time = self.df.loc[idx]
        label = self.cls2id[name]
        time_step = self.duration
        time_step = 5
        # if self.train:
        #     time_step = random.random() * (self.duration - 3) + 3
        #     # time_step = random.randint(3, self.duration)
            
        x, orig_sr = librosa.load(
            f"dataset/train_audio/{path}",
            sr=None,
            # sr=self.SR,
            offset=random.randint(0, max(time - 5, 0.)),
            duration=time_step,
        )
        if orig_sr != self.SR:
            x = librosa.resample(x, orig_sr=orig_sr, target_sr=self.SR)
        
        # rand_num = random.random()
        # if self.train and rand_num < .8:
        #     num_sample = random.randint(1, 4)
        #     x, label = self.mixup(x, label, num_sample)
        # elif self.train and rand_num < .85:
        #     x, label = self.cutmix(x, label)
            
        # if self.audio_transform:
        #     x = self.audio_transform(samples=x, sample_rate=self.SR)

        # x = self.pipeline(x)
        
#         if self.spec_transofrm:
#             x = self.spec_transofrm(x)
        
#         if self.transform:
#             x = self.transform(image=x)["image"]
        x = load_and_transform_audio_data(x)
        x = self.tensor_transform(x)
    
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
        
        # db_map = librosa.amplitude_to_db(mels, ref=np.max)
        # db_map = librosa.power_to_db(mels, ref=np.max)
        # db_map = (db_map + 80) / 80
        
        # pcen_map = pcen(mels).astype(np.float32)
        # a_db_map = librosa.amplitude_to_db(mels, ref=np.max) / 80 + 1
        # p_db_map = librosa.power_to_db(mels, ref=np.max) / 80 + 1
        # db_map = np.stack([pcen_map, a_db_map, p_db_map], axis=-1)
        
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
                duration=random.random() * (self.duration - 3) + 3,
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
            duration=random.random() * (self.duration - 3) + 3,
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

