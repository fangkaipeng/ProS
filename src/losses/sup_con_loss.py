from __future__ import print_function

import torch
import torch.nn as nn

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
def sup_con_loss(features, temperature=0.07, contrast_mode='all', base_temperature=0.07, labels=None, mask=None, device = None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device) # 对角线全1的矩阵
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1] # 4 batch = 10
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 40,50
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature) # 40 40
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # 40,1
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count) # 40, 40
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    ) # 对角线为0，其余为1
    mask = mask * logits_mask # 把对角线的1全变为0

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask # 除了对角线都有值
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # 每一行表示ancher和除自己以外的feature之和,然后logit中减去这个值

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # 40

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean() # 4,10 -> mean

    return loss

def soft_sup_con_loss(features, softlabels, hard_labels, temperature=0.07, base_temperature=0.07, device = None):
    """Compute loss for model. 
    Args:
        features: hidden vector of shape [bsz, hide_dim].
        soft_labels : hidden vector of shape [bsz, hide_dim].
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hard_labels = hard_labels.contiguous().view(-1, 1)
    # mask = torch.eq(hard_labels, hard_labels.T).float().to(device) # batch, batch
    
    # compute logits
    features_dot_softlabels = torch.div(torch.matmul(features, softlabels.T), temperature) # 
    predict = torch.argmax(features_dot_softlabels, 1)
    correct = (predict == hard_labels).sum().item()
    loss = torch.nn.functional.cross_entropy(features_dot_softlabels, hard_labels)
    # # for numerical stability
    # logits_max, _ = torch.max(features_dot_softlabels, dim=1, keepdim=True) # 40,1
    # logits = features_dot_softlabels - logits_max.detach()

    # logits = torch.exp(logits)
    # loss = torch.log((logits * mask).sum(1)) - torch.log((logits.sum(1) - (logits * mask).sum(1)))
    # loss = loss / mask.sum(1)
    # loss = -loss.mean()

    return loss, correct
