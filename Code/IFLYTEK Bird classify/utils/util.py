import torch
import torch.nn as nn

import numpy as np
from timm.layers.drop import DropPath


__all__ = ["get_param_groups"]
    
    
def get_param_groups(model, nowd_keys=()):
    para_groups, para_groups_dbg = {}, {}
    
    for name, para in model.named_parameters():
        if not para.requires_grad:
            continue  # frozen weights
        if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
            wd_scale, group_name = 0., 'no_decay'
        else:
            wd_scale, group_name = 1., 'decay'
        
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
            para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)
    
#     for g in para_groups_dbg.values():
#         g['params'] = pformat(', '.join(g['params']), width=200)
    
#     print(f'[get_ft_param_groups] param groups = \n{pformat(para_groups_dbg, indent=2, width=250)}\n')
    return list(para_groups.values())


def update_dropout_rate(model, rate=.0):
    """
    更改模型中的所有 Dropout 层的rate
    
    Args:
        model (nn.Module): 模型
        rate (float): dropout使用的值
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = rate
        elif isinstance(module, DropPath):
            module.drop_prob = rate


def drop_scheduler(drop_rate=.0, epochs=20, niter_per_ep=5, cutoff_epoch=5, mode="standard", schedule="constant"):
    """ dropout的调度值
    
    Args:
        drop_rate (float): dropout值
        epochs (int): epoch的次数
        niter_per_ep (int): 每个epoch有n个iter
        cutoff_epoch (int): epoch截断
        mode (str): 使用的模式
            standard: 全局使用dropout
            early: 前期使用
            late: 后期使用
        schedule (str): dropout率的调度
    """
    assert mode in ["standard", "early", "late"]
    if mode == "standard":
        return np.full(epochs * niter_per_ep, drop_rate)

    early_iters = cutoff_epoch * niter_per_ep
    late_iters = (epochs - cutoff_epoch) * niter_per_ep

    if mode == "early":
        assert schedule in ["constant", "linear"]
        if schedule == 'constant':
            early_schedule = np.full(early_iters, drop_rate)
        elif schedule == 'linear':
            early_schedule = np.linspace(drop_rate, 0, early_iters)
        final_schedule = np.concatenate((early_schedule, np.full(late_iters, 0)))

    elif mode == "late":
        assert schedule in ["constant"]
        early_schedule = np.full(early_iters, 0)
        final_schedule = np.concatenate((early_schedule, np.full(late_iters, drop_rate)))

    assert len(final_schedule) == epochs * niter_per_ep
    return final_schedule