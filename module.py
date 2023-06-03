import torch.nn as nn
import torch.nn.functional as F
import math


from global_configs import *


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, config):
        super(MHAtt, self).__init__()
        self.config = config

        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_merge = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.config.multi_head,
            int(self.config.hidden_size_head)
        ).transpose(1, 2)
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.config.multi_head,
            int(self.config.hidden_size_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.config.multi_head,
            int(self.config.hidden_size_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.config.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=config.hidden_size,
            mid_size=config.ff_size,
            out_size=config.hidden_size,
            dropout_r=config.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, config):
        super(SA, self).__init__()

        self.mhatt = MHAtt(config)
        self.ffn = FFN(config)
        self.dropout1 = nn.Dropout(config.dropout_r)
        self.norm1 = LayerNorm(config.hidden_size)

        self.dropout2 = nn.Dropout(config.dropout_r)
        self.norm2 = LayerNorm(config.hidden_size)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))
        return x


# ------------------------
# ----Cross  Attention ----
# ------------------------
class CA(nn.Module):
    def __init__(self, config):
        super(CA, self).__init__()

        self.mhatt1 = MHAtt(config)
        self.ffn = FFN(config)

        self.dropout1 = nn.Dropout(config.dropout_r)
        self.norm1 = LayerNorm(config.hidden_size)

        self.dropout2 = nn.Dropout(config.dropout_r)
        self.norm2 = LayerNorm(config.hidden_size)

        self.linear_1 = nn.Linear(TEXT_DIM, config.hidden_size)
        self.linear_2 = nn.Linear(config.hidden_size, TEXT_DIM)

    def forward(self, x, y, x_mask, y_mask):

        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, y, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x
