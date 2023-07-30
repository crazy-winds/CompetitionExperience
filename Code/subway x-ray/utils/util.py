import torch
import torch.nn as nn

import numpy as np
from timm.layers.drop import DropPath


def bbox_ioa(box1, box2, eps=1e-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are xywh
    box1:       np.array of shape(5)
    box2:       np.array of shape(nx5)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, w1, h1, *_ = box1
    b2_x1, b2_y1, w2, h2, *_ = box2.T
    b1_x2, b1_y2 = (b1_x1 + w1, b1_y1 + h1)
    b2_x2, b2_y2 = (b2_x1 + w2, b2_y1 + h2)

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area

    
    
def get_param_groups(model, nowd_keys=()):
    para_groups, para_groups_dbg = {}, {}
    assert isinstance(nowd_keys, (list, tuple))
    
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

class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        adv_step=1,
        platform=None
    ):
        """
        From https://www.kaggle.com/code/wht1996/feedback-nn-train
        
        Use:
            awp = AWP(model, optimizer)
            for x, y in dataloader:
                loss = model(x, y)
                loss.backward()
                awp.attack_backward(x, y)
                optimizer.step()
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.platform = platform

    def attack_backward(self, batch_inputs):
        if (self.adv_lr == 0):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step()
            adv_loss = self.model(batch_inputs, mode="loss")
            adv_loss = sum(adv_loss)
            self.optimizer.zero_grad()
            self.platform.backward(adv_loss)
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}