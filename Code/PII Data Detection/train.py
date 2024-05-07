#!/usr/bin/env python
# coding: utf-8

# In[199]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    DebertaV2Model,
    DebertaV2Config,
    AutoModelForTokenClassification,
)
from tokenizers import AddedToken

from timm.optim.adan import Adan
from timm.scheduler.cosine_lr import CosineLRScheduler

import json
import random
from itertools import chain
from collections import defaultdict

import gc
import tqdm
import texttable
from copy import deepcopy

from utils import (
    util,
    re_find,
    preprocess_data
)
from accelerate import Accelerator
from concurrent.futures import ThreadPoolExecutor

import spacy
nlp = spacy.load("en_core_web_sm")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # Parameters

# In[200]:


SEED = 3407
MODEL_NAME = "Debert-v3-Base"
PRETRAIN_MODEL = "deberta-v3-base/"
# MODEL_NAME = "Debert-v3-Large"
# PRETRAIN_MODEL = "deberta-v3-large/"

DATA_PATH = [
    "dataset/train.json",
    "dataset/moredata_dataset_fixed.json",
    # "dataset/moth.json",
    "dataset/Nicholas.json",
    # "dataset/Fake_data_1850_218.json",
    # "dataset/mpware_mixtral8x7b_v1.1-no-i-username.json",
]
VALID_RATE = .2
BATCH_SIZE = 1
NUM_WORKERS = 2
TRAIN_MAX_LEN = 768
VALID_MAX_LEN = 768
MAX_POS_EMBED = 3840

LR = 5e-5
WD = 1e-2
EPOCHS = 25
VALID_EPOCH = 1
LABEL_WEIGHT = [1.] * 13
LABEL_WEIGHT[12] = .93
# LABEL_WEIGHT[4] = 1.07
# LABEL_WEIGHT[10] = 1.07
O_THRESHOLD = .9

GRADIENT_ACCUMULAT = 8
PRINT_NAME = [
    "EMAIL",
    "ID_NUM",
    "NAME_STUDENT",
    "PHONE_NUM",
    "STREET_ADDRESS",
    "URL_PERSONAL",
    "USERNAME",
]
id2pos = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CONJ",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "SPACE",
]
pos2id = {p: i for i, p in enumerate(id2pos)}


# In[201]:


accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULAT)
device = accelerator.device

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# # Dataset

# In[202]:


tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL)
# tokenizer.add_tokens(AddedToken("\n", normalized=False))


# In[203]:


with open("dataset/train.json", "r") as f:
    id2label = sorted(list(set(chain(*[x["labels"] for x in json.load(f)]))))
    label2id = {l: i for i, l in enumerate(id2label)}


# In[204]:


def read_data(data_path, offset=0, reset_doc=True):
    """ 读取 JSON 数据集

    Args:
        data_path (List[str]): 所有数据集的路径
    """
    all_data = []
    for dp in data_path:
        with open(dp, "r") as f:
            all_data.append(json.load(f))
    
    all_data = sum(all_data, [])
    i = 0
    while i < len(all_data):
        # https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/468844
        if all([label == "O" for label in all_data[i]["labels"]]) and random.random() > .25:
        # if all([label == "O" for label in all_data[i]["labels"]]):
            all_data.pop(i)
        else: 
            i += 1
    
    # 重新编号，以防重复
    if reset_doc:
        for i in range(offset, len(all_data) + offset):
            all_data[i - offset]["document"] = i
        
    return all_data


# In[205]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def __len__(self):
        return self.length


# In[206]:


def random_choise(N):
    if N == 1:
        return 0
        
    weights = []
    middle = N // 2
    for i in range(N):
        weight = 1 - abs(i - (N-1) / 2) / ((N-1) / 2)
        weights.append(1 - weight)
        
    samples = random.choices(range(N), weights=weights)
    return samples[0]


# In[207]:


def train_collect_fn(batch):
    batch_text = []
    char2token_idx = []
    token2char_start_idx = []
    token2char_end_idx = []
    for item in batch:
        text = []
        c2Ti = []
        ti2Cs = []
        ti2Ce = []
        char_start_idx = 0
        char_end_idx = 0
        for idx, (token, whitespace) in enumerate(
            zip(
                item["tokens"],
                item["trailing_whitespace"],
            )
        ):
            ti2Cs.append(char_start_idx)
            
            text.append(token)
            c2Ti += [idx] * len(token)
            char_start_idx += len(token)
            char_end_idx += len(token)
            
            if whitespace:
                text.append(" ")
                c2Ti.append(-1)
                char_start_idx += 1
                char_end_idx += 1
            ti2Ce .append(char_end_idx)
                
        batch_text.append("".join(text))
        char2token_idx.append(c2Ti)
        token2char_start_idx.append(ti2Cs)
        token2char_end_idx.append(ti2Ce)

    char2token_idx = {item["document"]: char2token_idx[i] for i, item in enumerate(batch)}
    label = {item["document"]: item["labels"] for i, item in enumerate(batch)}

    input_ids = []
    atten_mask = []
    offset_map = []
    token_type_ids = []
    position_ids = []
    paragraph_offset = []
    document = []
    for batch_id, item in enumerate(batch):
        token_pos = item["pos"]
        doc_id = item["document"]
        children_paragraph = item["children_paragraph"]
        paper_token_offset = item["token_offset"]
        tokenized = tokenizer(
            batch_text[batch_id],
            truncation=False,
            return_offsets_mapping=True,
            max_length=None,
            padding=False,
        )
        # # for paragraph_id in range(len(children_paragraph)):
        # paragraph_id = random.randint(0, len(children_paragraph) - 1)
        # paragraph_offset_start, paragraph_offset_end = (
        #     len("".join(children_paragraph[:paragraph_id])),
        #     len("".join(children_paragraph[:paragraph_id + 1])),
        # )
        # paragraph_id = random.randint(0, len(paper_token_offset) - 1)
        paragraph_id = random_choise(len(paper_token_offset))
        # paragraph_id = epoch % len(paper_token_offset)
        # for paragraph_id in range(len(paper_token_offset)):
        paragraph_offset_start, paragraph_offset_end = (
            token2char_start_idx[batch_id][paper_token_offset[paragraph_id][0]],
            token2char_end_idx[batch_id][paper_token_offset[paragraph_id][1] - 1],
        )

        (
            paper_input_ids,
            paper_atten_mask,
            paper_offset_map,
        ) = (
            tokenized["input_ids"],
            tokenized["attention_mask"],
            tokenized["offset_mapping"],
        )
        # 处理 [UNK] 的特殊情况
        i = 0
        while i < len(paper_input_ids):
            if paper_input_ids[i] == 3:
                paper_input_ids.pop(i - 1)
                paper_atten_mask.pop(i - 1)
                paper_offset_map.pop(i - 1)
            else:
                i += 1
        tokenized_length = len(paper_input_ids)
        
        start_idx = 0
        end_idx = tokenized_length - 1
        while True:
            s, e = paper_offset_map[start_idx]
            # 特殊字符
            if s == 0 and e == 0:
                start_idx += 1
                continue
            if s >= paragraph_offset_start:
                if paper_offset_map[start_idx - 1][1] > paragraph_offset_start:
                    start_idx -= 1
                break
            else:
                start_idx += 1

        while True:
            s, e = paper_offset_map[end_idx]
            # 特殊字符
            if s == 0 and e == 0:
                end_idx -= 1
                continue
            if e <= paragraph_offset_end:
                break
            else:
                end_idx -= 1

        input_ids.append(paper_input_ids[start_idx: end_idx + 1])
        atten_mask.append(paper_atten_mask[start_idx: end_idx + 1])
        offset_map.append(paper_offset_map[start_idx: end_idx + 1])
        pos = []
        tti = []
        c2t = char2token_idx[doc_id]
        txt = batch_text[batch_id]
        for start, end in paper_offset_map[start_idx: end_idx + 1]:
            if start + end == 0:
                pos.append(MAX_POS_EMBED - 1)
                tti.append(pos2id["SPACE"])
                continue
            while start < len(txt) and txt[start].isspace():
                start += 1
            try:
                pos.append(c2t[start] % (MAX_POS_EMBED - 1))
                tti.append(token_pos[c2t[start]])
            except IndexError:
                tti.append(pos2id["SPACE"])
                pos.append(MAX_POS_EMBED - 1)

        token_type_ids.append(tti)
        # position_ids.append(pos)
        # position_ids.append(list(range(start_idx, end_idx + 1)))
        position_ids.append(list(range(end_idx - start_idx + 1)))
        paragraph_offset.append((paragraph_offset_start, paragraph_offset_end))
        document.append(item["document"])
    
    batch_sequence_length = min(max([len(input_ids[i]) for i in range(len(input_ids))]), TRAIN_MAX_LEN + 0)
    for batch_id in range(len(input_ids)):
        if len(input_ids[batch_id]) == batch_sequence_length:
            continue
        elif len(input_ids[batch_id]) > batch_sequence_length:
            input_ids[batch_id] = input_ids[batch_id][:batch_sequence_length]
            atten_mask[batch_id] = atten_mask[batch_id][:batch_sequence_length]
            offset_map[batch_id] = offset_map[batch_id][:batch_sequence_length]
            token_type_ids[batch_id] = token_type_ids[batch_id][:batch_sequence_length]
            position_ids[batch_id] = position_ids[batch_id][:batch_sequence_length]
        else:
            difference_length = batch_sequence_length - len(input_ids[batch_id])
            input_ids[batch_id] += [tokenizer.pad_token_id] * difference_length
            atten_mask[batch_id] += [0] * difference_length
            offset_map[batch_id] += [(0, 0)] * difference_length
            position_ids[batch_id] += list(range(position_ids[batch_id][-1] + 1, position_ids[batch_id][-1] + 1 + difference_length))
            token_type_ids[batch_id] += [pos2id["SPACE"]] * difference_length
            # position_ids[batch_id] += [MAX_POS_EMBED - 1] * difference_length

    batch_text = {item["document"]: batch_text[i] for i, item in enumerate(batch)}
    batch_label = []
    for batch_id in range(len(input_ids)):
        doc_id = document[batch_id]
        cur_label = label[doc_id]
        cur_text = batch_text[doc_id]
        cur_c2t = char2token_idx[doc_id]
        bl = []
        for token_start, token_stop in offset_map[batch_id]:
            # 特殊 token
            if (token_start == 0 and token_stop == 0):
                bl.append(label2id["O"])
                continue

            while cur_c2t[token_start] == -1 or cur_text[token_start].isspace():
                token_start += 1
            
            label_type = cur_label[cur_c2t[token_start]]
            bl.append(label2id[label_type])
        batch_label.append(bl)
        
    label_token = set()
    label_label = set()
    for item in batch:
        doc_id = item["document"]
        for idx, (t, l) in enumerate(zip(item["tokens"], item["labels"])):
            if l != "O":
                label_token.add((doc_id, idx, t))
                label_label.add((doc_id, idx, l))

    # return {
    #     "text": batch_text,
    #     "input_ids": [torch.LongTensor(i)[None] for i in input_ids],
    #     "atten_mask": [torch.LongTensor(i)[None] for i in atten_mask],
    #     "position_ids": [torch.LongTensor(i)[None] for i in position_ids],
    #     "label": [torch.LongTensor(i)[None] for i in batch_label],
    #     # "label_type_counts": torch.FloatTensor(batch_count),
    #     "evaluate_label": {
    #         "token_set": label_token,
    #         "label_set": label_label,
    #     },
    #     "offset_map": offset_map,
    #     "raw_label": label,
    #     "raw_token": {item["document"]: item["tokens"] for i, item in enumerate(batch)},
    #     "document": document,
    #     "char2token_idx":char2token_idx,
    #     "paragraph_offset": paragraph_offset,
    # }
    return {
        "text": batch_text,
        "input_ids": torch.LongTensor(input_ids),
        "atten_mask": torch.LongTensor(atten_mask),
        "token_type_ids": torch.LongTensor(token_type_ids),
        "position_ids": torch.LongTensor(position_ids),
        "label": torch.LongTensor(batch_label),
        "evaluate_label": {
            "token_set": label_token,
            "label_set": label_label,
        },
        "offset_map": offset_map,
        "raw_label": label,
        "raw_token": {item["document"]: item["tokens"] for i, item in enumerate(batch)},
        "document": document,
        "char2token_idx":char2token_idx,
        "paragraph_offset": paragraph_offset,
    }


# In[208]:


def test_collect_fn(batch):
    batch_text = []
    char2token_idx = []
    token2char_start_idx = []
    token2char_end_idx = []
    for item in batch:
        text = []
        c2Ti = []
        ti2Cs = []
        ti2Ce = []
        char_start_idx = 0
        char_end_idx = 0
        for idx, (token, whitespace) in enumerate(
            zip(
                item["tokens"],
                item["trailing_whitespace"],
            )
        ):
            ti2Cs.append(char_start_idx)
            
            text.append(token)
            c2Ti += [idx] * len(token)
            char_start_idx += len(token)
            char_end_idx += len(token)
            
            if whitespace:
                text.append(" ")
                c2Ti.append(-1)
                char_start_idx += 1
                char_end_idx += 1
            ti2Ce .append(char_end_idx)
                
        batch_text.append("".join(text))
        char2token_idx.append(c2Ti)
        token2char_start_idx.append(ti2Cs)
        token2char_end_idx.append(ti2Ce)

    char2token_idx = {item["document"]: char2token_idx[i] for i, item in enumerate(batch)}
    
    input_ids = []
    atten_mask = []
    offset_map = []
    token_type_ids = []
    position_ids = []
    paragraph_offset = []
    document = []
    for batch_id, item in enumerate(batch):
        children_paragraph = item["children_paragraph"]
        paper_token_offset = item["token_offset"]
        doc_id = item["document"]
        tokenized = tokenizer(
            batch_text[batch_id],
            truncation=False,
            return_offsets_mapping=True,
            max_length=None,
            padding=False,
        )
        (
            paper_input_ids,
            paper_atten_mask,
            paper_offset_map,
        ) = (
            tokenized["input_ids"],
            tokenized["attention_mask"],
            tokenized["offset_mapping"],
        )
        # 处理 [UNK] 的特殊情况
        i = 0
        while i < len(paper_input_ids):
            if paper_input_ids[i] == 3:
                paper_input_ids.pop(i - 1)
                paper_atten_mask.pop(i - 1)
                paper_offset_map.pop(i - 1)
            else:
                i += 1
        tokenized_length = len(paper_input_ids)
        
        # # for paragraph_id in range(len(children_paragraph)):
        # paragraph_id = random.randint(0, len(children_paragraph) - 1)
        # paragraph_offset_start, paragraph_offset_end = (
        #     len("".join(children_paragraph[:paragraph_id])),
        #     len("".join(children_paragraph[:paragraph_id + 1])),
        # )
        # paragraph_id = random.randint(0, len(paper_token_offset) - 1)
        for paragraph_id in range(len(paper_token_offset)):
            paragraph_offset_start, paragraph_offset_end = (
                token2char_start_idx[batch_id][paper_token_offset[paragraph_id][0]],
                token2char_end_idx[batch_id][paper_token_offset[paragraph_id][1] - 1],
            )
            start_idx = 0
            end_idx = tokenized_length - 1
            while True:
                s, e = paper_offset_map[start_idx]
                # 特殊字符
                if s == 0 and e == 0:
                    start_idx += 1
                    continue
                if s >= paragraph_offset_start:
                    if paper_offset_map[start_idx - 1][1] > paragraph_offset_start:
                        start_idx -= 1
                    break
                else:
                    start_idx += 1

            while True:
                s, e = paper_offset_map[end_idx]
                # 特殊字符
                if s == 0 and e == 0:
                    end_idx -= 1
                    continue
                if e <= paragraph_offset_end:
                    break
                else:
                    end_idx -= 1

            input_ids.append(paper_input_ids[start_idx: end_idx + 1])
            atten_mask.append(paper_atten_mask[start_idx: end_idx + 1])
            offset_map.append(paper_offset_map[start_idx: end_idx + 1])

            pos = []
            tti = []
            c2t = char2token_idx[doc_id]
            txt = batch_text[batch_id]
            for start, end in paper_offset_map[start_idx: end_idx + 1]:
                if start + end == 0:
                    pos.append(MAX_POS_EMBED - 1)
                    tti.append(pos2id["SPACE"])
                    continue
                while start < len(txt) and txt[start].isspace():
                    start += 1
                try:
                    pos.append(c2t[start] % (MAX_POS_EMBED - 1))
                    tti.append(item["pos"][c2t[start]])
                except IndexError:
                    tti.append(pos2id["SPACE"])
                    pos.append(MAX_POS_EMBED - 1)

            token_type_ids.append(tti)
            # position_ids.append(pos)
            # position_ids.append(list(range(start_idx, end_idx + 1)))
            position_ids.append(list(range(end_idx - start_idx + 1)))
            paragraph_offset.append((paragraph_offset_start, paragraph_offset_end))
            document.append(item["document"])
    
    batch_sequence_length = max([len(input_ids[i]) for i in range(len(input_ids))])
    for batch_id in range(len(input_ids)):
        if len(input_ids[batch_id]) == batch_sequence_length:
            continue
        difference_length = batch_sequence_length - len(input_ids[batch_id])
        input_ids[batch_id] += [tokenizer.pad_token_id] * difference_length
        atten_mask[batch_id] += [0] * difference_length
        offset_map[batch_id] += [(0, 0)] * difference_length
        position_ids[batch_id] += list(range(position_ids[batch_id][-1] + 1, position_ids[batch_id][-1] + 1 + difference_length))
        token_type_ids[batch_id] += [pos2id["SPACE"]] * difference_length
        # position_ids[batch_id] += [MAX_POS_EMBED - 1] * difference_length
    
    label_token = set()
    label_label = set()
    for item in batch:
        doc_id = item["document"]
        for idx, (t, l) in enumerate(zip(item["tokens"], item["labels"])):
            if l != "O":
                label_token.add((doc_id, idx, t))
                label_label.add((doc_id, idx, l))
        
    return {
        "text": {item["document"]: batch_text[i] for i, item in enumerate(batch)},
        "input_ids": torch.LongTensor(input_ids),
        "atten_mask": torch.LongTensor(atten_mask),
        "token_type_ids": torch.LongTensor(token_type_ids),
        "position_ids": torch.LongTensor(position_ids),
        "evaluate_label": {
            "token_set": label_token,
            "label_set": label_label,
        },
        "offset_map": offset_map,
        "raw_token": {item["document"]: item["tokens"] for i, item in enumerate(batch)},
        "document": document,
        "char2token_idx": char2token_idx,
        "paragraph_offset": paragraph_offset,
    }


# In[209]:


train_data = read_data(DATA_PATH)
random.shuffle(train_data)
valid_data = []
for i in range(int(len(train_data) * VALID_RATE)):
    valid_data.append(train_data.pop())
    # valid_data.append(train_data[i])

# train_data = read_data([DATA_PATH[0]], reset_doc=False)
# valid_data = [d for d in train_data if d["document"] % 4 == 0]
# train_data = [d for d in train_data if d["document"] % 4 != 0]
# if len(DATA_PATH) > 1:
#     train_data = train_data + read_data(DATA_PATH[1:], int(1e5))
    
for i in range(len(train_data)):
    paper, token_offset = preprocess_data.cut_paper(train_data[i], TRAIN_MAX_LEN)
    train_data[i].update({
        "children_paragraph": paper,
        "token_offset": token_offset,
    })
    train_data[i]["tokens"] = [token.replace("\n", "[SEP]") for token in train_data[i]["tokens"]]
    
for i in range(len(valid_data)):
    paper, token_offset = preprocess_data.cut_paper(valid_data[i], VALID_MAX_LEN)
    valid_data[i].update({
        "children_paragraph": paper,
        "token_offset": token_offset,
    })
    valid_data[i]["tokens"] = [token.replace("\n", "[SEP]") for token in valid_data[i]["tokens"]]

# 词性分析
def process_item(item):
    item["pos"] = [pos2id[token.pos_] for token in nlp(item["full_text"])]

with ThreadPoolExecutor(max_workers=None) as executor:
    executor.map(process_item, train_data)

with ThreadPoolExecutor(max_workers=None) as executor:
    executor.map(process_item, valid_data)

# In[210]:


# 正则匹配 `电话号码`和 `邮箱地址`
train_re_email, train_re_phone = re_find.find_phone_and_email(train_data)
train_re_set = defaultdict(set)
for email in train_re_email:
    train_re_set[email["document"]].add((
        email["document"],
        email["token"],
        email["label"],
    ))
    
for phone in train_re_phone:
    train_re_set[phone["document"]].add((
        phone["document"],
        phone["token"],
        phone["label"],
    ))
    
valid_re_email, valid_re_phone = re_find.find_phone_and_email(valid_data)
valid_re_set = defaultdict(set)
for email in valid_re_email:
    valid_re_set[email["document"]].add((
        email["document"],
        email["token"],
        email["label"],
    ))
    
for phone in valid_re_phone:
    valid_re_set[phone["document"]].add((
        phone["document"],
        phone["token"],
        phone["label"],
    ))


# In[211]:


train_dataloader = torch.utils.data.DataLoader(
    Dataset(train_data),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    collate_fn=train_collect_fn
)
valid_dataloader = torch.utils.data.DataLoader(
    Dataset(valid_data),
    batch_size=max(BATCH_SIZE // 2, 1),
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=test_collect_fn
)


# In[228]:


# batch = next(iter(train_dataloader))
# (
#     input_ids,
#     atten_mask,
#     pos_ids,
#     y,
    
#     documents,
#     label_set,
#     offset_map,
#     doc2raw_token,
#     char2token_idx,
# ) = (
#     batch["input_ids"],
#     batch["atten_mask"],
#     batch["position_ids"],
#     batch["label"],
    
#     batch["document"],
#     batch["evaluate_label"]["label_set"],
#     batch["offset_map"],
#     batch["raw_token"],
#     batch["char2token_idx"],
# )
# token_str = list(doc2raw_token.values())[0]
# char2token = list(char2token_idx.values())[0]
# doc_id = documents[0]

# for i, v in enumerate(batch["input_ids"][0]):
#     if v == 3:
#         print(i)


# In[229]:


# for i, (token, (s, e), l) in enumerate(zip(
#     tokenizer.tokenize(tokenizer.decode(batch["input_ids"][0])),
#     offset_map[0],
#     y[0],
# )):
#     while s < len(char2token) and (
#         char2token[s] == -1 or 
#         token_str[char2token[s]].isspace()
#     ):
#         s += 1
        
#     raw_token_idx = char2token[s]
#     print(
#         f"{str(raw_token_idx).ljust(20)}\t",
#         f"{token.ljust(20)}\t",
#         f"{token_str[raw_token_idx].ljust(20)}\t"
#         f"{id2label[int(l)].ljust(20)}\t"
#     )


# In[ ]:


del train_data, valid_data
gc.collect()


# # Model

from collections.abc import Sequence
from transformers.modeling_outputs import BaseModelOutput


class DebertaV3Ner(DebertaV2Model):
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        pos_ids=None,
        output_attentions=False,
        layer_mask=None,
        *args,
        **kwargs,
    ):
        if layer_mask is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=pos_ids,
                output_attentions=output_attentions,
                **kwargs,
            )
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        device = input_ids.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            mask=attention_mask,
        )
        # # # #  Encoder # # # # #
        """
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        """
        query_states = None
        if layer_mask is None:
            layer_mask = [
                self.encoder.get_attention_mask(attention_mask)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            layer_mask = [
                self.encoder.get_attention_mask(mask)
                for mask in layer_mask
            ]
        relative_pos = self.encoder.get_rel_pos(embedding_output, query_states, None)
        all_attentions = ()
        if isinstance(embedding_output, Sequence):
            next_kv = embedding_output[0]
        else:
            next_kv = embedding_output
        rel_embeddings = self.encoder.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.encoder.layer):
            if self.encoder.gradient_checkpointing and self.training:
                output_states = self.encoder._gradient_checkpointing_func(
                    layer_module.__call__,
                    next_kv,
                    layer_mask[i],
                    query_states,
                    relative_pos,
                    rel_embeddings,
                    output_attentions,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    layer_mask[i],
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.encoder.conv is not None:
                output_states = self.encoder.conv(embedding_output, output_states, attention_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)
        # # # # #  End # # # # # #

        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=None,
            attentions=all_attentions,
        )


def compute_actual_lens_rev(attn_mask, device, num_heads, g_dropout=.3):
    max_len = attn_mask.size(-1)
    actual_len = torch.sum(attn_mask, dim=-1)
    drop_nums = ((1- g_dropout) * actual_len).long()  # [N,]  ---> [N, 12, L, L]
    drop_nums_onehot = torch.eye(max_len).to(device)[drop_nums]  # [N, L]
    drop_nums_onehot = drop_nums_onehot.unsqueeze(1).unsqueeze(2).expand([-1, num_heads, max_len, -1])
    return drop_nums_onehot

def generate_mask_rev(grad_t, extend_attn_mask, attn_mask, device, num_heads, g_dropout=.3, keep_rate=.9):
    # 计算要mask的数量
    max_len = attn_mask.size(-1)
    drop_nums_onehot = compute_actual_lens_rev(attn_mask, device, num_heads, g_dropout)
    # 排序
    grad_p = grad_t + (1 - extend_attn_mask) * 100 # [B, H, L, L]
    sorted_grad = torch.sort(grad_p, dim=-1)[0] # [B, H, L, L]
    st_grad = torch.sum(drop_nums_onehot * sorted_grad, dim=-1) # [B, H, L]
    st_grad = st_grad.unsqueeze(-1).expand([-1, -1, -1, max_len]) # [B, H, L, L]
    grad_masks = (1 - torch.ge(grad_p, st_grad).long())  # 反向mask
    # random select
    sampler_rate = keep_rate * torch.ones_like(grad_masks).to(device)
    sampler_masks = torch.bernoulli(sampler_rate)

    total_masks = ((grad_masks + sampler_masks) >= 1).long() * extend_attn_mask

    return total_masks


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        config = DebertaV2Config.from_pretrained(PRETRAIN_MODEL)
        config.update({
            # "max_position_embeddings": MAX_POS_EMBED,
            # "position_biased_input": True,
            "hidden_dropout_prob": .1,
            "attention_probs_dropout_prob": .0,
            "num_labels": num_classes,
            "type_vocab_size": len(pos2id),
        })
        self.model = DebertaV3Ner.from_pretrained(
            PRETRAIN_MODEL,
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.model.embeddings.word_embeddings.weight.requires_grad_(False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.loss_fct = nn.CrossEntropyLoss()

    def call_forward(self, input_ids, attention_mask, output_attentions=False, **kwargs):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs
        )

        sequence_output = output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if output_attentions:
            return logits, output.attentions
        else:
            return logits
        
    def forward(self, input_ids, atten_mask, token_type_ids=None, pos_ids=None, do_mask=True):
        if self.training and do_mask:
            encoder_attn_mask = atten_mask.unsqueeze(1).unsqueeze(2).expand([-1, self.model.config.num_attention_heads, atten_mask.size(-1), -1])
            encoder_attn_mask_list = [encoder_attn_mask for i in range(self.model.config.num_hidden_layers)]
            
            output, attentions = self.call_forward(
                input_ids=input_ids,
                attention_mask=atten_mask,
                token_type_ids=token_type_ids,
                pos_ids=pos_ids,
                output_attentions=True,
                layer_mask=None,
            )
            B, N, C = output.shape

            p_labels = output.argmax(-1)
            active_mask = atten_mask.bool().view(-1)
            active_logits = output.view(-1, C)[active_mask]
            active_labels = p_labels.view(-1)[active_mask]
            p_loss = self.loss_fct(active_logits, active_labels)

            for layer_id in range(self.model.config.num_hidden_layers):
                attentions[layer_id].retain_grad()

                (grad, ) = torch.autograd.grad(p_loss, attentions[layer_id], retain_graph=True)
                grad_masks = generate_mask_rev(
                    -grad,
                    encoder_attn_mask, 
                    atten_mask,
                    device=atten_mask.device,
                    num_heads=self.model.config.num_attention_heads,
                    g_dropout=.8,
                    keep_rate=.9,
                )
                encoder_attn_mask_list[layer_id] = grad_masks.long()
            layer_mask = torch.stack(encoder_attn_mask_list, dim=0)

            output = self.call_forward(
                input_ids=input_ids,
                attention_mask=atten_mask,
                token_type_ids=token_type_ids,
                pos_ids=pos_ids,
                layer_mask=layer_mask,
            )

            return output
            
        else:
            return self.call_forward(
                input_ids=input_ids,
                attention_mask=atten_mask,
                token_type_ids=token_type_ids,
                pos_ids=pos_ids,
            )


class CustomCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction = "none"
        
    def forward(self, x, y, atten_mask=None, counts=None):
        B, L, C = x.shape
        x = x.reshape(-1, C)
        y = y.reshape(-1)
        loss = super().forward(x, y)
        loss = loss.reshape(B, L)
        if atten_mask is not None:
            loss = loss * atten_mask

        ohem_loss, _ = loss.sort(dim=-1, descending=True)
        ohem_loss = (ohem_loss[:, :3] * .5).sum(-1).mean()
        return loss.sum(-1).mean() + ohem_loss


# In[ ]:


model = Model(len(label2id))


# In[ ]:


# optimizer = Adan(
optimizer = torch.optim.AdamW(
    util.get_param_groups(model),
    lr=LR,
    weight_decay=WD,
)
lr_scheduler = CosineLRScheduler(
    optimizer=optimizer,
    t_initial=EPOCHS,
    warmup_t=EPOCHS // 6,
    warmup_lr_init=LR / 10,
)
criterion = CustomCrossEntropy(
    weight=torch.tensor(LABEL_WEIGHT, dtype=torch.float, device=device)
)
# criterion = nn.CrossEntropyLoss(
#     weight=torch.tensor([12.] * (len(label2id) - 1) + [1.], dtype=torch.float, device=device)
# )


# In[ ]:


model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
    model, 
    optimizer,
    train_dataloader,
    valid_dataloader,
    lr_scheduler,
)


# # Trainer

# In[ ]:


class Evaluate:
    def __init__(
        self,
        tp=0,
        fp=0,
        fn=0,
    ):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        
    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self
    
    def __add__(self, other):
        return Evaluate(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )
    
    @property
    def precision(self) -> float:
        if self.tp == 0:
            return 0.
        return self.tp / (self.tp + self.fp)
        
    @property
    def recall(self) -> float:
        if self.tp == 0:
            return 0.
        return self.tp / (self.tp + self.fn)
    
    def f_score(self, beta=5.):
        p = self.precision
        r = self.recall
        if p == 0 or r == 0:
            return 0.
        
        return (
            (1 + beta ** 2) * 
            p * r /
            (beta ** 2 * p + r)
        )


# In[ ]:


def token_mapping(pred, offset, char2token_idx, documents, raw_tokens):
    """ 将bert tokenizer化后的 token 还原，并筛除 "O"
    """
    result = set()
    hash_map = set()
    for batch_idx in range(len(pred)):
        doc_id = documents[batch_idx]
        char2token = char2token_idx[doc_id]
        token_str = raw_tokens[doc_id]
        # p_off = paragraph_offset[batch_idx]
        for pred_class, (start_idx, end_idx) in \
            zip(pred[batch_idx], offset[batch_idx]):
            if start_idx == 0 and end_idx == 0:
                continue
                
            pred_type = id2label[int(pred_class)]
            if pred_type == "O":
                continue
                
            while start_idx < len(char2token) and (
                char2token[start_idx] == -1 or 
                token_str[char2token[start_idx]].isspace()
            ):
                start_idx += 1
                
            while start_idx >= len(char2token):
                start_idx -= 1
                
            raw_token_idx = char2token[start_idx]
            if (doc_id, raw_token_idx) not in hash_map:
                result.add((doc_id, raw_token_idx, pred_type))
                hash_map.add((doc_id, raw_token_idx))
            
    return result


# In[ ]:


def compute_one_batch(
    pred,
    documents,
    target_set,
    offset_map,
    raw_tokens,
    char2token_idx,
    # paragraph_offset,
    cur_state=None,
    re_result=None,
):
    """ 计算一个 batch 的 tp、fp、fn
    
    Args:
        pred (np.ndarray): 经过softmax的预测输出，shape(B, L)
        documents (List[int]): 当前子句所对应的论文
        target_set (set[Tuple]): Label 的论文id、token索引及标签的元组集合
        offset_map (List[Tuple[int]]): tokenizer的偏移
        raw_tokens (Dict[int, List]): 每篇论文的id所对应的原始所有tokens
        char2token_idx (Dict[int, List]): 每篇论文的id所对应char到tokens的索引
        paragraph_offset (List[Tuple(int)]): 子句所对应原论文字符的起始位置和结束位置
    """
    if cur_state is None:
        cur_state = defaultdict(Evaluate)
    target_set = deepcopy(target_set)
    pred_set = token_mapping(pred, offset_map, char2token_idx, documents, raw_tokens)
    
    for i, pred_tuple in enumerate(pred_set):
        pred_type = pred_tuple[-1][2:]
        if pred_tuple in target_set:
            cur_state[pred_type].tp += 1
            target_set.remove(pred_tuple)
        else:
            cur_state[pred_type].fp += 1
    
    for target_tuple in target_set:
        tar_type = target_tuple[-1][2:]
        cur_state[tar_type].fn += 1
            
    return cur_state


# In[ ]:


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:


def train_one_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    dataloader = tqdm.tqdm(
        dataloader,
        desc="Trainer: ",
        disable=not accelerator.is_main_process
    )
    
    losses = 0
    class_score = defaultdict(Evaluate)
    for batch in dataloader:
        with accelerator.accumulate(model):
            (
                input_ids,
                atten_mask,
                token_type_ids,
                pos_ids,
                y,

                documents,
                label_set,
                offset_map,
                doc2raw_token,
                char2token_idx,
                # paragraph_offset,
            ) = (
                batch["input_ids"],
                batch["atten_mask"],
                batch["token_type_ids"],
                batch["position_ids"],
                batch["label"],

                batch["document"],
                batch["evaluate_label"]["label_set"],
                batch["offset_map"],
                batch["raw_token"],
                batch["char2token_idx"],
                # batch["paragraph_offset"],
            )
            batch_length = len(input_ids)
            if batch_length <= BATCH_SIZE:
                out = model(input_ids, atten_mask, token_type_ids, pos_ids)
                loss = criterion(out, y, atten_mask)
            else:
                out = []
                loss = 0
                for i in range(0, batch_length, BATCH_SIZE):
                    out.append(
                        model(
                            input_ids[i],
                            atten_mask[i],
                            pos_ids[i],
                        )
                    )
                    loss = loss + criterion(
                        out[-1],
                        y[i],
                        atten_mask[i]
                    )
                
                # out = []
                # for i in range(0, batch_length, BATCH_SIZE):
                #     out.append(
                #         model(
                #             input_ids[i: i + BATCH_SIZE],
                #             atten_mask[i: i + BATCH_SIZE],
                #             pos_ids[i: i + BATCH_SIZE],
                #         )
                #     )
                # out = torch.cat(out)
            # loss = criterion(out, y, atten_mask)
            losses += loss.item()

            accelerator.backward(loss)
            optimizer.step()
            out = [o.argmax(-1).detach().cpu().numpy() for o in out]
            # https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/470978
            # out = output.softmax(-1).detach().cpu().numpy()
            # out[:, -1:] = np.where(out[:, -1:] > O_THRESHOLD, out[:, -1:], 0)
            # out = np.argmax(out, axis=-1)

            # 将长度之外的值除去
            # out[~atten_mask.cpu().numpy().astype(bool)] = label2id["O"]
            all_out = accelerator.gather_for_metrics(
                (
                    out,
                    documents,
                    label_set,
                    offset_map,
                    doc2raw_token,
                    char2token_idx,
                )
            )
            for i in range(0, len(all_out), 6):
                out = all_out[i]
                doc = all_out[i + 1]
                label = all_out[i + 2]
                offset = all_out[i + 3]
                r_token = all_out[i + 4]
                c2t = all_out[i + 5]
                compute_one_batch(out, doc, label, offset, r_token, c2t, class_score, train_re_set)
                
            optimizer.zero_grad()
            del (
                out,
                all_out,
                input_ids,
                atten_mask,
                pos_ids,
                y,
            )
            clear_memory()
            
    total_score = sum(class_score.values(), Evaluate())

    tabel_print = texttable.Texttable(max_width=120)
    tabel_print.add_row(["Classes", "Summary", *PRINT_NAME])
    tabel_print.add_row([
        "T F5 Score",
        total_score.f_score(5),
        *[class_score[name].f_score(5) for name in PRINT_NAME]
    ])
    tabel_print.add_row([
        "T Precision",
        total_score.precision,
        *[class_score[name].precision for name in PRINT_NAME]
    ])
    tabel_print.add_row([
        "T Recall",
        total_score.recall,
        *[class_score[name].recall for name in PRINT_NAME]
    ])
    accelerator.print(tabel_print.draw())
    accelerator.print(
        f"TP: {total_score.tp}\t"
        f"FP: {total_score.fp}\t"
        f"FN: {total_score.fn}\n"
    )
    
    return losses / len(dataloader)


# In[ ]:


@torch.no_grad()
def valid_one_epoch(model, dataloader):
    model.eval()
    dataloader = tqdm.tqdm(
        dataloader,
        desc="Valid: ",
        disable=not accelerator.is_main_process
    )
    
    class_score = defaultdict(Evaluate)
    for batch in dataloader:
        (
            input_ids,
            atten_mask,
            token_type_ids,
            pos_ids,
            
            documents,
            label_set,
            offset_map,
            doc2raw_token,
            char2token_idx,
            # paragraph_offset,
        ) = (
            batch["input_ids"],
            batch["atten_mask"],
            batch["token_type_ids"],
            batch["position_ids"],
            
            batch["document"],
            batch["evaluate_label"]["label_set"],
            batch["offset_map"],
            batch["raw_token"],
            batch["char2token_idx"],
            # batch["paragraph_offset"],
        )
        batch_length = input_ids.shape[0]
        if batch_length <= BATCH_SIZE:
            output = model(input_ids, atten_mask, token_type_ids, pos_ids)
            out = output.argmax(-1).detach().cpu().numpy()
        else:
            output = []
            for i in range(0, batch_length, BATCH_SIZE):
                output.append(
                    model(
                        input_ids[i: i + BATCH_SIZE],
                        atten_mask[i: i + BATCH_SIZE],
                        token_type_ids[i: i + BATCH_SIZE],
                        pos_ids[i: i + BATCH_SIZE],
                    ).argmax(-1).detach().cpu().numpy()
                )
            out = np.concatenate(output)
        # https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/470978
        # out = output.softmax(-1).detach().cpu().numpy()
        # out[:, -1:] = np.where(out[:, -1:] > O_THRESHOLD, out[:, -1:], 0)
        # out = np.argmax(out, axis=-1)
        
        # if accelerator.is_local_main_process:
        # 将长度之外的值除去
        out[~atten_mask.cpu().numpy().astype(bool)] = label2id["O"]
        all_out = accelerator.gather_for_metrics(
            (
                out,
                documents,
                label_set,
                offset_map,
                doc2raw_token,
                char2token_idx,
                # paragraph_offset,
            )
        )
        for i in range(0, len(all_out), 6):
            out = all_out[i]
            doc = all_out[i + 1]
            label = all_out[i + 2]
            offset = all_out[i + 3]
            r_token = all_out[i + 4]
            c2t = all_out[i + 5]
            compute_one_batch(out, doc, label, offset, r_token, c2t, class_score, valid_re_set)
    
    total_score = sum(class_score.values(), Evaluate())
    
    tabel_print = texttable.Texttable(max_width=120)
    tabel_print.add_row(["Classes", "Summary", *PRINT_NAME])
    tabel_print.add_row([
        "V F5 Score",
        total_score.f_score(5),
        *[class_score[name].f_score(5) for name in PRINT_NAME]
    ])
    tabel_print.add_row([
        "V F1 Score",
        total_score.f_score(1),
        *[class_score[name].f_score(1) for name in PRINT_NAME]
    ])
    tabel_print.add_row([
        "V Precision",
        total_score.precision,
        *[class_score[name].precision for name in PRINT_NAME]
    ])
    tabel_print.add_row([
        "V Recall",
        total_score.recall,
        *[class_score[name].recall for name in PRINT_NAME]
    ])
    accelerator.print(tabel_print.draw())
    accelerator.print(
        f"TP: {total_score.tp}\t"
        f"FP: {total_score.fp}\t"
        f"FN: {total_score.fn}\n"
    )
    
    return total_score, class_score


# In[ ]:


min_loss = float("inf")
max_score = 0

for epoch in range(EPOCHS):
     # # # # #
    # Train #
    # # # # #
    loss = train_one_epoch(model, train_dataloader, optimizer, criterion, epoch)
    lr_scheduler.step(epoch + 1)
    accelerator.print(
        f"Epoch {epoch} - Loss: {loss :.4f}\t"
        f"lr: {optimizer.param_groups[0]['lr']*1e4 :.4f}\t"
        # f"lr: {optimizer.param_groups[0]['show_lr']*1e4 :.4f}\t"
    )
    if min_loss > loss:
        min_loss = loss
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
    clear_memory()
    
    # # # # #
    # Valid #
    # # # # #
    if epoch % VALID_EPOCH == 0:
        total_score, class_score = valid_one_epoch(model, valid_dataloader)
        s = total_score.f_score(5)
        if s >= max_score:
            torch.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    # "ema": ema_model.module.state_dict()
                },
                f"work_dir/{MODEL_NAME}_best.pt"
            )
            max_score = s
        # accelerator.print(f"Valid Epoch {epoch} - Score: {s :.2f}%\tMax Score: {max_score :.2f}%")
    clear_memory()

