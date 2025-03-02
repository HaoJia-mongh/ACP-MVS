import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .position_encoding import PositionEncodingSine


class DepthHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, scale=False):
        super(DepthHead, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x_d, act_fn=torch.tanh):
        out = self.conv2(self.relu(self.conv1(x_d)))
        return act_fn(out)


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class ProjectionInputDepth(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)

        self.convd = nn.Conv2d(64+hidden_dim, out_chs - 1, 3, padding=1)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, depth, cost):
        # print(cost.size())
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        dfm = F.relu(self.convd1(depth))
        dfm = F.relu(self.convd2(dfm))
        cor_dfm = torch.cat([cor, dfm], dim=1)

        out_d = F.relu(self.convd(cor_dfm))
        if self.training and self.dropout is not None:
            out_d = self.dropout(out_d)
        return torch.cat([out_d, depth], dim=1)


class UpMaskNet(nn.Module):
    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, feat):
        mask = .25 * self.mask(feat)
        return mask


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        new_values = self.inner_attention(
            queries,
            keys,
            values,
        ).view(N, L, -1)

        return self.out_projection(new_values)


class LinearAttLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None):
        super(LinearAttLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        inner_attention = LinearAttention()
        attention = AttentionLayer(inner_attention, d_model, n_heads, d_keys, d_values)

        self.attention = attention
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, context, encoded_volume):

        cga_volume = encoded_volume + self.gamma * self.attention(context, context, encoded_volume)

        return cga_volume


class CGA(nn.Module):
    def __init__(self):
        super(CGA, self).__init__()

        self.linear_att = LinearAttLayer(64, 1)
        self.pos_encoding = PositionEncodingSine(64)

    def forward(self, context=None, encoded_volume=None):

        _, _, H, _ = context.shape
        context = einops.rearrange(self.pos_encoding(context), 'n c h w -> n (h w) c')
        encoded_volume = einops.rearrange(encoded_volume, 'n c h w -> n (h w) c')
        cga_volume = self.linear_att(context, encoded_volume)

        return einops.rearrange(cga_volume, 'n (h w) c -> n c h w', h=H)


class CoarsestUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=256, ratio=8, context_dim=64 , UpMask=False):
        super(CoarsestUpdateBlockDepth, self).__init__()

        self.encoder = ProjectionInputDepth(cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.depth_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim+self.encoder.out_chs)
        self.depth_head = DepthHead(hidden_dim, hidden_dim=hidden_dim, scale=False)
        self.UpMask = UpMask
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

        self.cga = CGA()

    def forward(self, net, depth_cost_func, depth, context, seq_len=4, scale_inv_depth=None):
        depth_list = []
        mask_list = []
        for i in range(seq_len):

            # TODO detach()
            # inv_depth  torch.Size([12, 1, 64, 80])
            depth = depth.detach()
            encoded_volume = self.encoder(depth, depth_cost_func(scale_inv_depth(depth)[1]))
            # input_features  torch.Size([12, 64, 64, 80])
            # attention  torch.Size([12, 2, 5120, 5120])
            cga_volume = self.cga(context, encoded_volume)

            # motion_features_global  torch.Size([12, 64, 64, 80])
            # context  torch.Size([12, 64, 64, 80])
            inp = torch.cat([context, encoded_volume, cga_volume], dim=1)

            net = self.depth_gru(net, inp)

            delta_depth = self.depth_head(net)

            depth = depth + delta_depth
            depth_list.append(depth)
            if self.UpMask and i == seq_len - 1 :
                mask = .25 * self.mask(net)
                mask_list.append(mask)
            else:
                mask_list.append(depth)
        return net, mask_list, depth_list