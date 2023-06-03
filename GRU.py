import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, args):
        super(GRUCell, self).__init__()
        self.in2hid_w = nn.ModuleList([nn.Linear(768, 768) for _ in range(3)])
        self.hid2hid_w = nn.ModuleList([nn.Linear(768, 768) for _ in range(3)])

    def forward(self, x, hid):
        # 重置门，重置hid再与x结合
        r = torch.sigmoid(self.in2hid_w[0](x) + self.hid2hid_w[0](hid))
        z = torch.sigmoid(self.in2hid_w[1](x) + self.hid2hid_w[1](hid))
        n = torch.tanh(self.in2hid_w[2](x) + torch.mul(r, self.hid2hid_w[2](hid)))
        next_hid = (1 - z) * hid + z * n

        return next_hid
