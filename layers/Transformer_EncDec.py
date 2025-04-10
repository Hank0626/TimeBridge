import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers.Embed import denormalization
from layers.SelfAttention_Family import ASAttentionLayer, AsymAttention


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", group=None, router=None,
                 stable=True):
        super(EncoderLayer, self).__init__()

        self.group = group
        self.stable = stable
        if self.group is not None:
            self.inner_attn = attention
            self.inter_attn = copy.deepcopy(attention)
        self.router = router
        if self.router is not None:
            self.sender = attention
            self.receiver = copy.deepcopy(attention)
            self.router = nn.Parameter(torch.randn(router, d_model))

        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.mattention = copy.deepcopy(attention)
        self.sattention = copy.deepcopy(attention)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.group is not None:
            B, ch, dim = x.shape
            new_x = rearrange(x, 'B (num_g group) p -> (B num_g) group p', group=self.group)
            new_x, attn = self.inner_attn(
                new_x, new_x, new_x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
            new_x = rearrange(new_x, '(B num_g) group p -> (B group) num_g p', B=B)
            new_x, attn = self.inter_attn(
                new_x, new_x, new_x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
            new_x = new_x.reshape(B, ch, dim)
        elif self.router is not None:
            B, ch, dim = x.shape
            batch_router = self.router.unsqueeze(0).repeat(B, 1, 1)
            buffer, attn = self.sender(
                batch_router, x, x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
            new_x, attn = self.receiver(
                x, buffer, buffer,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )

        else:
            qk = x
            with torch.no_grad():
                if self.stable:
                    qk = PeriodNorm(x)
            new_x, attn = self.attention(
                qk, qk, x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class AsymEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", c_in=None, layer_norm=True):
        super(AsymEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.proj1 = nn.Linear(d_model, d_ff)
        self.proj2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model) if layer_norm else nn.BatchNorm2d(c_in)
        self.norm2 = nn.LayerNorm(d_model) if layer_norm else nn.BatchNorm2d(c_in)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, x_mark=None, attn_type=0):
        new_x, attn = self.attention(x, x_mark, attn_type)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.proj1(y)))
        y = self.dropout(self.proj2(y))

        out = self.norm2(x + y)
        return out, attn


class AsymEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(AsymEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, x_mark=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, x_mark)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x.reshape(x.shape[0], x.shape[1], -1), attns


class WUEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(WUEncoder, self).__init__()
        self.norm = norm_layer
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x_l, x_h):
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x_l, x_h = attn_layer(x_l, x_h)

        return self.norm(x_l), x_h


class WUEncoderLayer(nn.Module):
    def __init__(
            self, c_in, num_g, n_heads,
            d_model, d_ff=None, kernel=None, dropout=0.1, activation="relu"
    ):
        super(WUEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = ASAttentionLayer(d_model=d_model, c_in=c_in,
                                          num_g=num_g, n_heads=n_heads, kernel_size=kernel)

        self.proj1 = nn.Linear(d_model, d_ff)
        self.proj2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, xl, xh=None):
        new_xl, xh = self.attention(xl, xh)
        xl = self.norm1(xl + self.dropout(new_xl))
        yl = self.dropout(self.activation(self.proj1(xl)))
        xl = self.norm2(xl + self.dropout(self.proj2(yl)))

        return xl, xh


class DualEncoderLayer(nn.Module):
    def __init__(self, num_p, d_model, d_ff=None, n_heads=8,
                 c_in=862, dropout=0.1, activation="relu"):
        super(DualEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.Attention = AsymAttention(d_model, n_heads, c_in + 4)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, x_mark=None):
        # B, ch, dim = x.shape
        new_x = self.Attention(x)
        x = self.norm1(x + self.dropout(new_x))
        x = self.norm2(x + self.FFN(x))
        return x, None


class ChannelEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(ChannelEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, x_mark=None, attn_type=None):
        # x [B, L, D]
        attns = []
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, x_mark, attn_type=i if attn_type is None else attn_type)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TemporalEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(TemporalEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, x_mark=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, x_mark)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class ConDecoder(nn.Module):
    def __init__(self, d_model=512, pred_len=720, alpha=0.3):
        super(ConDecoder, self).__init__()

        self.xc_proj = nn.Linear(d_model, pred_len)
        self.xt_proj = nn.Linear(d_model, pred_len)

    def forward(self, x, msc, mst):
        xc, xt = torch.unbind(x, dim=-2)
        xc, xt = self.xc_proj(xc), self.xt_proj(xt)
        xc, xt = denormalization(xc, *msc), denormalization(xt, *mst)
        return xc + xt


class GroupEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(GroupEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, k=None):
        for i, attn_layer in enumerate(self.attn_layers):
            x = attn_layer(x, k[i])
        return x


class GroupEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", c_in=None, layer_norm=True):
        super(GroupEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.proj1 = nn.Linear(d_model, d_ff)
        self.proj2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model) if layer_norm else nn.BatchNorm2d(c_in)
        self.norm2 = nn.LayerNorm(d_model) if layer_norm else nn.BatchNorm2d(c_in)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, k=None):
        new_x = self.attention(x, k)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.proj1(y)))
        y = self.dropout(self.proj2(y))

        out = self.norm2(x + y)
        return out


class PureEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", c_in=None, layer_norm=True):
        super(PureEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.proj1 = nn.Linear(d_model, d_ff)
        self.proj2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, k=None):
        new_x = self.attention(x, k)
        x = x + self.dropout(new_x)
        y = x

        y = self.dropout(self.activation(self.proj1(y)))
        y = self.dropout(self.proj2(y))

        out = x + y
        return out


class SeasonalLayer(nn.Module):
    def __init__(self, attention, configs):
        super(SeasonalLayer, self).__init__()

        d_ff = configs.d_ff
        d_model = configs.d_model

        self.attention = attention

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if configs.activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        b, c, n, d = x.shape

        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x, attn = self.attention(
            new_x, new_x, new_x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        new_x = rearrange(new_x, '(b n) c d-> b c n d', b=b)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), attn


class DSGEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 c_in=None, layer_norm=True, configs=None):
        super(DSGEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.num_p = configs.num_p
        total_p = self.num_p * (configs.ratio + 1)
        self.multi_patch = nn.Linear(total_p, total_p, bias=False)
        self.attention = attention

        self.proj1 = nn.Linear(d_model, d_ff)
        self.proj2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model) if layer_norm else nn.BatchNorm2d(c_in)
        self.norm2 = nn.LayerNorm(d_model) if layer_norm else nn.BatchNorm2d(c_in)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, k=None):
        # if k is None:
        new_x = self.attention(x, k)
        new_x = self.multi_patch(new_x.transpose(-1, -2)).transpose(-1, -2)
        # else:
        #     new_x = self.s_attention(x, k)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.proj1(y)))
        y = self.dropout(self.proj2(y))

        out = self.norm2(x + y)
        return out


class TSEncoder(nn.Module):
    def __init__(self, attn_layers):
        super(TSEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        return x, attns


class IntAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, stable_len=8, attn_map=False,
                 dropout=0.1, activation="relu", stable=True, enc_in=None):
        super(IntAttention, self).__init__()
        self.stable = stable
        self.stable_len = stable_len
        self.attn_map = attn_map
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.temporal_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def temporal_attn(self, x):
        b, c, n, d = x.shape
        new_x = x.reshape(-1, n, d)

        qk = new_x
        if self.stable:
            with torch.no_grad():
                qk = PeriodNorm(new_x, self.stable_len)
        new_x = self.attention(qk, qk, new_x)[0]
        new_x = new_x.reshape(b, c, n, d)
        return new_x


def PeriodNorm(x, period_len=6):
    if len(x.shape) == 3:
        x = x.unsqueeze(-2)
    b, c, n, t = x.shape
    x_patch = [x[..., period_len - 1 - i:-i + t] for i in range(0, period_len)]
    x_patch = torch.stack(x_patch, dim=-1)

    mean = x_patch.mean(4)
    mean = F.pad(mean.reshape(b * c, n, -1),
                 mode='replicate', pad=(period_len - 1, 0)).reshape(b, c, n, -1)
    out = x - mean
    return out.squeeze(-2)


class PatchSampling(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 in_p=30, out_p=4, stable=False, stable_len=8):
        super(PatchSampling, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.in_p = in_p
        self.out_p = out_p
        self.stable = stable
        self.stable_len = stable_len

        self.attention = attention
        self.conv1 = nn.Conv1d(
            self.in_p, self.out_p, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv1d(
            self.out_p + 1, self.out_p, 1, 1, 0, bias=False)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.down_attn(x)
        y = x = self.norm1(new_x)

        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def down_attn(self, x):
        b, c, n, d = x.shape
        x = x.reshape(-1, n, d)
        new_x = self.conv1(x)
        new_x = self.conv2(torch.cat(
            [new_x, x.mean(-2, keepdim=True)], dim=-2)) + new_x
        new_x = self.attention(new_x, x, x)[0] + self.dropout(new_x)
        return new_x.reshape(b, c, -1, d)


class CointAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, axial=True, stable_len=8,
                 dropout=0.1, activation="relu", stable=True, enc_in=None, ):
        super(CointAttention, self).__init__()

        self.stable = stable
        self.stable_len = stable_len
        d_ff = d_ff or 4 * d_model

        self.axial_func = axial
        self.attention1 = attention
        self.attention2 = copy.deepcopy(attention)

        self.num_rc = math.ceil((enc_in + 4) ** 0.5)
        self.pad_ch = nn.ConstantPad1d(
            (0, self.num_rc ** 2 - (enc_in + 4)), 0)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.axial_func is True:
            new_x = self.axial_attn(x)
        else:
            new_x = self.full_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def axial_attn(self, x):
        b, c, n, d = x.shape

        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x = (self.pad_ch(new_x.transpose(-1, -2))
                 .transpose(-1, -2).reshape(-1, self.num_rc, d))
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b r) c d -> (b c) r d', r=self.num_rc)
        new_x = self.attention2(new_x, new_x, new_x)[0] + new_x

        new_x = rearrange(new_x, '(b n c) r d -> b (r c) n d', b=b, n=n)
        return new_x[:, :c, ...]

    def full_attn(self, x):
        b, c, n, d = x.shape
        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b n) c d -> b c n d', b=b, n=n)
        return new_x[:, :c, :]
