import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, attn_map=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.attn_map = attn_map
        self.alpha = nn.Parameter(torch.rand(1))
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, long_term=True):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn_map = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(attn_map)
        if self.attn_map is True:
            heat_map = attn_map[:, ...].max(1)[0]
            heat_map = torch.clamp_max(heat_map, 0.15)
            # heat_map = torch.softmax(heat_map, -1)
            for b in range(heat_map.shape[0]):
                # for c in range(heat_map.shape[1]):
                h_map = heat_map[b, ...].detach().cpu().numpy()
                # plt.savefig(heat_map, f'{b} sample {c} channel')
                plt.figure(figsize=(10, 8), dpi=200)
                plt.imshow(h_map, cmap='Reds', interpolation='nearest')
                plt.colorbar()

                # 设置X轴和Y轴的标签为黑体文字
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = ['Times New Roman']
                plt.xlabel('Key Channel', fontsize=14)
                plt.ylabel('Query Channel', fontsize=14)

                # 设置标题
                # plt.title('Long-Term Correlations', fontdict={'weight': 'bold'}, fontsize=16, color='green')

                plt.tight_layout()
                plt.savefig(f'./stable map/{b}_sample.png')
                # plt.savefig(f'./non_stable map/{b}_sample.png')
                plt.close()
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if self.inner_attention is None:
            return self.out_projection(self.value_projection(values)), None
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut q_proj=k_proj
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class AsymAttentionLayer(nn.Module):
    def __init__(self, d_model, num_g, num_p=1, n_heads=8, Asym=True, clip=1):
        super(AsymAttentionLayer, self).__init__()

        assert d_model % clip == 0
        self.Asym = Asym
        self.clip = clip
        self.num_g = num_g
        self.num_p = num_p
        self.n_heads = n_heads
        self.d_model = d_model

        self.inner_group_attn = FullAttention(False, output_attention=False)
        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.v1 = nn.Linear(d_model, d_model)

        self.inter_group_attn = FullAttention(False, output_attention=False)
        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(d_model, d_model)
        self.v2 = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, x_mark=None):

        if self.Asym is True:
            return self.Asym_attn(x)
        else:
            return self.all_attn(x)

    def Asym_attn(self, x):
        B, ch, num_p, dim = x.shape

        x = rearrange(x, 'B (num_g g) num_p p -> (B num_g num_p) g p', num_g=self.num_g)
        q = self.q1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.inner_group_attn(q, k, v, None)
        x = rearrange(x, '(B num_g num_p) g num_h p -> (B g num_p) num_g (num_h p)', B=B, num_g=self.num_g)

        q = self.q2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.inter_group_attn(q, k, v, None)
        x = rearrange(x, '(B g num_p) num_g num_h p -> B (num_g g) num_p (num_h p)', B=B, num_p=num_p)
        return self.out(x), attn

    def all_attn(self, x):
        B, ch, num_p, dim = x.shape
        x = rearrange(x, 'B ch num_p p -> (B num_p) ch p')

        q = self.q1(x).view(x.shape[0], ch, self.n_heads, -1)
        k = self.k1(x).view(x.shape[0], ch, self.n_heads, -1)
        v = self.v1(x).view(x.shape[0], ch, self.n_heads, -1)
        x, attn = self.inner_group_attn(q, k, v, None)

        x = rearrange(x, '(B num_p) ch num_h dim -> B ch num_p (num_h dim)', B=B)
        return self.out(x), attn


class ASAttentionLayer(nn.Module):
    def __init__(self, d_model, num_g, n_heads=8, c_in=7, kernel_size=None):
        super(ASAttentionLayer, self).__init__()

        self.num_g = num_g
        self.n_heads = n_heads

        self.conv = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel_size, padding=kernel_size // 2, groups=c_in),
            nn.BatchNorm1d(c_in)
        ) if kernel_size is not None else nn.Identity()

        self.inner_group_attn = FullAttention(False, output_attention=False)
        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.v1 = nn.Linear(d_model, d_model)

        self.inter_group_attn = FullAttention(False, output_attention=False)
        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(d_model, d_model)
        self.v2 = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, xl, xh):
        B, ch, dim = xl.shape

        xl = rearrange(xl, 'B (num_g g) p -> (B num_g) g p', num_g=self.num_g)
        q = self.q1(xl).view(xl.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k1(xl).view(xl.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v1(xl).view(xl.shape[0], -1, self.n_heads, dim // self.n_heads)
        xl, attn = self.inner_group_attn(q, k, v, None)

        xl = rearrange(xl, '(B num_g) g num_h p -> (B g) num_g (num_h p)', B=B)
        q = self.q2(xl).view(xl.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k2(xl).view(xl.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v2(xl).view(xl.shape[0], -1, self.n_heads, dim // self.n_heads)
        xl, attn = self.inter_group_attn(q, k, v, None)

        xl = rearrange(xl, '(B g) num_g num_h p -> B (num_g g) (num_h p)', B=B)
        return self.out_projection(xl), self.conv(xh)


class AsymAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, num_g=None, c_in=866):
        super(AsymAttention, self).__init__()

        group = math.ceil(c_in ** 0.5)
        self.num_g = math.ceil(c_in / group) if num_g is None else num_g

        self.n_heads = n_heads
        self.d_model = d_model

        self.inner_group_attn = FullAttention(False, output_attention=False)
        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.v1 = nn.Linear(d_model, d_model)
        self.inter_group_attn = FullAttention(False, output_attention=False)
        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(d_model, d_model)
        self.v2 = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, ch, dim = x.shape

        x = rearrange(x, 'B (num_g g) p -> (B num_g) g p', num_g=self.num_g)
        q = self.q1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.inner_group_attn(q, k, v, None)

        x = rearrange(x, '(B num_g) g num_h p -> (B g) num_g (num_h p)', B=B)
        q = self.q2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.inter_group_attn(q, k, v, None)

        x = rearrange(x, '(B g) num_g num_h p -> B (num_g g) (num_h p)', B=B)
        return self.out(x)


class AllAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super(AllAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        self.attn = FullAttention(False, output_attention=False)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, ch, dim = x.shape

        q = self.q(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.attn(q, k, v, None)

        return self.out(x.view(B, ch, dim))


class AttentionMixer(nn.Module):
    def __init__(self, d_model, num_g, num_p=1, n_heads=8, Asym=True, c_in=None,
                 temporal=False, dropout=0.1, seq_len=720):
        super(AttentionMixer, self).__init__()

        self.Asym = Asym
        self.num_g = num_g
        self.group = c_in // num_g
        self.num_p = num_p
        self.n_heads = n_heads
        self.d_model = d_model

        self.temporal = temporal
        self.attn = FullAttention(False, output_attention=False)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out1 = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.inner_group_attn = FullAttention(False, output_attention=False)
        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.v1 = nn.Linear(d_model, d_model)
        self.inter_group_attn = FullAttention(False, output_attention=False)
        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(d_model, d_model)
        self.v2 = nn.Linear(d_model, d_model)
        self.out2 = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, x_mark=None, attn_type=0):
        if self.Asym:
            if x.shape[-2] == 1:
                x = self.channel_mixer(x, x_mark)
            else:
                if self.temporal is False:
                    x = self.channel_mixer(x, x_mark) if attn_type % 2 == 0 else self.segment_mixer(x, x_mark)
                else:
                    xc = self.channel_mixer(x, x_mark) if attn_type % 2 == 0 else self.segment_mixer(x, x_mark)
                    xt = self.temporal_mixer(x)
                    x = xt + xc
        else:
            x = self.All_Mixer(x, x_mark)
        return x, None

    def channel_mixer(self, x, x_mark=None):
        B, ch, num_p, dim = x.shape

        x = x.view(B, self.num_g, -1, num_p, dim)
        x = torch.cat([x, x_mark.unsqueeze(1).repeat(1, self.num_g, 1, 1, 1)], dim=2) if x_mark is not None else x
        x = rearrange(x, 'B num_g g num_p p -> (B num_g) (g num_p) p')
        q = self.q1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.inner_group_attn(q, k, v, None)
        x = rearrange(x, '(B num_g) (g num_p) num_h p -> B num_g g num_p (num_h p)', B=B, num_p=num_p)
        if x_mark is not None:
            x = x[:, :, :-4, ...]

        x = torch.cat([x, x_mark.unsqueeze(2).repeat(1, 1, self.group, 1, 1)], dim=1) if x_mark is not None else x
        x = rearrange(x, 'B num_g g num_p p -> (B g) (num_g num_p) p')
        q = self.q2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.inter_group_attn(q, k, v, None)
        x = rearrange(x, '(B g) (num_g num_p) num_h p -> B num_g g num_p (num_h p)', B=B, num_p=num_p)
        x = x[:, :-4, ...].reshape(B, ch, num_p, dim) if x_mark is not None else x.reshape(B, ch, num_p, dim)
        return self.out2(x)

    def segment_mixer(self, x, x_mark=None):
        B, ch, num_p, dim = x.shape

        x = x.view(B, self.num_g, -1, num_p, dim)
        x = torch.cat([x, x_mark.unsqueeze(1).repeat(1, self.num_g, 1, 1, 1)], dim=2) if x_mark is not None else x
        x = rearrange(x, 'B num_g g num_p p -> (B num_g num_p) g p')
        q = self.q1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v1(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.inner_group_attn(q, k, v, None)
        x = rearrange(x[:, :-4, ...] if x_mark is not None else x,
                      '(B num_g num_p) g num_h p -> B num_g g num_p (num_h p)', B=B, num_p=num_p)

        x = torch.cat([x, x_mark.unsqueeze(2).repeat(1, 1, self.group, 1, 1)], dim=1) if x_mark is not None else x
        x = rearrange(x, 'B num_g g num_p p -> (B g num_p) num_g p')
        q = self.q2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v2(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.inter_group_attn(q, k, v, None)
        x = rearrange(x[:, :-4, ...] if x_mark is not None else x,
                      '(B g num_p) num_g num_h p -> B (num_g g) num_p (num_h p)', B=B, num_p=num_p)
        return self.out2(x)

    def temporal_mixer(self, x):
        B, ch, num_p, dim = x.shape

        x = x.reshape(-1, num_p, dim)
        q = self.q(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.attn(q, k, v, None)
        x = rearrange(x, '(B ch) num_p num_h p -> B ch num_p (num_h p)', B=B)
        return self.out1(x)

    def All_Mixer(self, x, x_mark=None):
        B, ch, num_p, dim = x.shape

        x = torch.cat([x, x_mark], dim=1) if x_mark is not None else x
        x = rearrange(x, 'B ch num_p p -> B (ch num_p) p')
        q = self.q(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        k = self.k(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        v = self.v(x).view(x.shape[0], -1, self.n_heads, dim // self.n_heads)
        x, attn = self.attn(q, k, v, None)
        x = rearrange(x, 'B (ch num_p) num_h p -> B ch num_p (num_h p)', num_p=self.num_p)
        x = x[:, :-4, ...] if x_mark is not None else x
        return self.out1(x)



class TSMixer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(TSMixer, self).__init__()

        self.attention = attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, res=False, attn=None):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads

        q = self.q_proj(q).reshape(B, L, H, -1)
        k = self.k_proj(k).reshape(B, S, H, -1)
        v = self.v_proj(v).reshape(B, S, H, -1)

        out, attn = self.attention(
            q, k, v,
            res=res, attn=attn
        )
        out = out.view(B, L, -1)

        return self.out(out), attn


class ResAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, scale=None, attn_map=False, nst=False):
        super(ResAttention, self).__init__()

        self.nst = nst
        self.scale = scale
        self.attn_map = attn_map
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, res=False, attn=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn_map = torch.softmax(scale * scores, dim=-1)
        if self.attn_map is True:
            heat_map = attn_map.reshape(32, -1, H, L, S)
            for b in range(heat_map.shape[0]):
                for c in range(heat_map.shape[1]):
                    h_map = heat_map[b, c, 0, ...].detach().cpu().numpy()
                    # plt.savefig(heat_map, f'{b} sample {c} channel')

                    plt.figure(figsize=(10, 8), dpi=200)
                    plt.imshow(h_map, cmap='Reds', interpolation='nearest')
                    plt.colorbar()

                    # 设置X轴和Y轴的标签为黑体文字
                    plt.rcParams['font.family'] = 'serif'
                    plt.rcParams['font.serif'] = ['Times New Roman']
                    plt.xlabel('Key Time Patch', fontsize=14)
                    plt.ylabel('Query Time Patch', fontsize=14)
                    plt.tight_layout()
                    if self.nst is True:
                        plt.savefig(f'./time map/{b}_sample_{c}_channel.png')
                    else:
                        plt.savefig(f'./stable time map/{b}_sample_{c}_channel.png')
                    # 关闭当前图形窗口
                    plt.close()
        A = self.dropout(attn_map)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A
