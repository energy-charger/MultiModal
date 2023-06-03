from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F


class SCL(nn.Module):
    def __init__(self, temperature=0.07):
        super(SCL, self).__init__()
        self.temperature = temperature

    def forward(self, feature_1, feature_2):
        device = (torch.device('cuda')
                  if feature_1.is_cuda
                  else torch.device('cpu'))
        # feature_1 = F.softmax(feature_1, dim=1)
        # feature_2 = F.softmax(feature_2, dim=1)
        # 负余弦相似度
        similarity_matrix = F.cosine_similarity(feature_1.unsqueeze(1), feature_2.unsqueeze(0), dim=2).to(
            device)
        n = feature_1.shape[0]
        mask_dui_jiao_0 = torch.eye(n, n).to(device)
        similarity_matrix = similarity_matrix * mask_dui_jiao_0
        loss_1 = torch.sum(similarity_matrix, dim=1)
        loss_1 = -loss_1
        loss = torch.sum(loss_1)/n

        return loss
