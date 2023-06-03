from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_1, feature_2, fusion_feature, labels=True, mask=None):
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
        device = (torch.device('cuda')
                  if fusion_feature.is_cuda
                  else torch.device('cpu'))

        similarity_matrix = F.cosine_similarity(fusion_feature.unsqueeze(1), fusion_feature.unsqueeze(0), dim=2).to(
            device)
        n = labels.shape[0]
        mask = torch.ones_like(similarity_matrix) * (labels.expand(n, n).eq(labels.expand(n, n).t()))
        mask.to(device)
        mask_no_sim = (torch.ones_like(mask) - mask).to(device)
        mask_dui_jiao_0 = (torch.ones(n, n) - torch.eye(n, n)).to(device)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        similarity_matrix = similarity_matrix * mask_dui_jiao_0
        sim = mask * similarity_matrix
        no_sim = similarity_matrix - sim
        no_sim_sum = torch.sum(no_sim, dim=1)
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend
        loss = torch.div(sim, sim_sum).to(device)
        loss = mask_no_sim + loss + torch.eye(n, n).to(device)
        # 接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  # 求-log
        mask_loss = mask * mask_dui_jiao_0
        mask_loss = torch.sum(mask_loss, dim=1)
        mask_2 = (mask_loss > 0).int()
        mask_loss = torch.ones_like(mask_loss)-mask_2 + mask_loss
        loss = torch.sum(torch.sum(loss, dim=1) / mask_loss) / n
        return loss
