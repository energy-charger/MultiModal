from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F

"""
单模态对比学习，模态对齐，保留模态间共信息
"""


class UnimodalConLoss(nn.Module):
    def __init__(self, temperature=0.7):
        super(UnimodalConLoss, self).__init__()
        self.temperature = temperature

    def uniLoss(self, feature1, feature2, feature3, labels=True):
        device = (torch.device('cuda')
                  if feature1.is_cuda
                  else torch.device('cpu'))

        feature_T = torch.cat([feature1, feature1], dim=0)
        feature_F = torch.cat([feature2, feature3], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        similarity_matrix = F.cosine_similarity(feature_T.unsqueeze(1), feature_F.unsqueeze(0), dim=2).to(
            device)
        n = labels.shape[0]
        mask = torch.ones_like(similarity_matrix) * (labels.expand(n, n).eq(labels.expand(n, n).t()))
        mask.to(device)
        mask_no_sim = (torch.ones_like(mask) - mask).to(device)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        sim = mask * similarity_matrix
        no_sim = similarity_matrix - sim
        no_sim_sum = torch.sum(no_sim, dim=1)
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend
        loss = torch.div(sim, sim_sum).to(device)
        loss = mask_no_sim + loss + torch.eye(n, n).to(device)
        # 接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  # 求-log
        mask_loss = mask
        mask_loss = torch.sum(mask_loss, dim=1)
        mask_2 = (mask_loss > 0).int()
        mask_loss = torch.ones_like(mask_loss) - mask_2 + mask_loss
        loss_T = torch.sum(torch.sum(loss, dim=1) / mask_loss) / (2 * n)

        return loss_T

    def forward(self, feature_V, feature_A, feature_T, labels=True, mask=None):
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
        loss_T = self.uniLoss(feature_T, feature_A, feature_V, labels)
        loss_A = self.uniLoss(feature_A, feature_T, feature_V, labels)
        feature_V = self.uniLoss(feature_V, feature_T, feature_A, labels)
        return (loss_T + loss_A + feature_V) / 3
