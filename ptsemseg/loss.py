import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.utils import *
from ptsemseg.lovasz_losses import *


def cross_entropy2d(input, target, weight=None, size_average=True, ignore_index=250):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsqueeze(1)
        target = F.interpolate(target, size=(h, w), mode='nearest')
        target = target.squeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear', align_corners=True)
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=ignore_index,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum().float()
    return loss


def seg_loss(input, target, weight=None, size_average=True, scale_weight=1.0, lambda_ce=1.0, lambda_lv=1.0):
    ce_loss = cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)
    lv_loss = lovasz_softmax(F.softmax(input, dim=1), target, only_present=False, per_image=True, ignore=250)

    loss = scale_weight * (lambda_ce * ce_loss + lambda_lv * lv_loss)
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None, lambda_ce=1.0, lambda_lv=1.0):
    if not isinstance(input, tuple):
        return seg_loss(input, target, weight=weight, size_average=size_average, lambda_ce=lambda_ce, lambda_lv=lambda_lv)

    n_inp = len(input)
    # Auxiliary training for PSPNet [1.0, 0.4]
    if scale_weight is None: # scale_weight: torch tensor type
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp, device=torch.device('cuda')), torch.arange(n_inp, device=torch.device('cuda')).float())

    loss = 0.0
    for i in range(n_inp):
        if isinstance(input[i], tuple):
            n_j = len(input[i])
            for j in range(n_j):
                loss = loss + seg_loss(input[i][j], target, weight=weight, size_average=size_average, scale_weight=scale_weight[i], lambda_ce=lambda_ce, lambda_lv=lambda_lv) / n_j
        else:
            loss = loss + seg_loss(input[i], target, weight=weight, size_average=size_average, scale_weight=scale_weight[i], lambda_ce=lambda_ce, lambda_lv=lambda_lv)
    return loss
