from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F


class ProtoConLoss(nn.Module):
    def __init__(self, temperature=0.7):
        super(ProtoConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, fusion_feature, labels=True, mask=None):
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

        u = torch.zeros([labels.shape[0], fusion_feature.shape[1]]).to(device)  # 原型数据
        for i in range(labels.shape[0]):
            if u[i].equal(torch.zeros(fusion_feature.shape[1]).to(device)):
                num = 0
                for j in range(labels.shape[0]):
                    if labels[i] == labels[j]:
                        u[i] += fusion_feature[j]
                        num += 1
                u[i] = u[i] / num
                for k in range(labels.shape[0]):
                    if labels[k] == labels[i]:
                        u[k] = u[i]

        similarity_matrix = F.cosine_similarity(fusion_feature.unsqueeze(1), u.unsqueeze(0), dim=2).to(device)
        n = labels.shape[0]
        mask_dui_jiao_0 = torch.eye(n, n).to(device)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        no_sim_sum = torch.sum(similarity_matrix, dim=1)
        sim = similarity_matrix * mask_dui_jiao_0
        similarity_matrix = torch.sum(sim, dim=1)

        loss = torch.div(similarity_matrix, no_sim_sum).to(device)

        # 接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  # 求-log

        loss = torch.sum(loss) / n
        return loss
