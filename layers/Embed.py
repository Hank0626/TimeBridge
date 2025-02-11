import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in logs space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class Group_DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,
                 channel=None, group=None):
        super(Group_DataEmbedding_inverted, self).__init__()
        self.channel_pad = nn.ConstantPad1d((0, group - channel % group), 0) \
            if channel % group > 0 else nn.Identity()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.channel_pad(x)
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_disentangle(nn.Module):
    def __init__(self, c_in, p_in, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_disentangle, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}

        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # self.pos_embedding = nn.Sequential(
        #     nn.Conv1d(freq_map[freq], p_in, 1, 1, 0),
        #     nn.Linear(seq_len, d_model),
        # )
        self.pos_dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x, x_mark = x.permute(0, 2, 1), x_mark.permute(0, 2, 1)
        # x, x_mark = self.value_embedding(x), self.pos_embedding(x_mark)
        #
        # return self.dropout(x), self.pos_dropout(x_mark)

        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)

        return self.dropout(x), None


class WTPatchEmbed(nn.Module):
    def __init__(self, patch, seq_len, d_model, dropout=0.1):
        super(WTPatchEmbed, self).__init__()

        self.dwt = DWT1DForward(wave='db2', J=1, mode='zero')
        self.idwt = DWT1DInverse(wave='db2', J=1, mode='zero')

        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.pos_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)

        return self.dropout(x), None


class AsymEmbedding(nn.Module):
    def __init__(
            self, seq_len, d_model,
            group, c_in, dropout=0.1, patch=1
    ):
        super(AsymEmbedding, self).__init__()

        self.group = group
        self.patch = patch

        self.padding_c = nn.ConstantPad1d((0, group - c_in % group), 0) if c_in % group > 0 else nn.Identity()
        self.padding_t = nn.ConstantPad1d((0, patch - seq_len % patch), 0) if seq_len % patch > 0 else nn.Identity()

        self.latent_proj = nn.Linear(patch, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = torch.cat([x, x_mark], dim=1)
        x = self.padding_t(x)
        x = self.padding_c(x.transpose(-1, -2)).transpose(-1, -2)

        x = rearrange(x, 'B ch (num_p p) -> B ch num_p p', p=self.patch)
        x = self.latent_proj(x)
        return self.dropout(x)


class WUEmbedding(nn.Module):
    def __init__(self, group, c_in, seq_len, d_model, drop_rate=0.1):
        super(WUEmbedding, self).__init__()

        self.group = group
        self.padding_c = nn.ConstantPad1d((0, group - c_in % group), 0) \
            if c_in % group > 0 \
            else nn.Identity()

        self.latent_proj = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(drop_rate)
        )

    def forward(self, x, x_mark=None):
        x = torch.cat([x, x_mark], dim=-1)
        x = self.padding_c(x).transpose(-1, -2)

        return self.latent_proj(x)


def normalization(x, mean=None, std=None):
    if mean is not None and std is not None:
        return (x - mean) / std
    mean = x.mean(-1, keepdim=True).detach()
    x = x - mean
    std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
    x /= std
    return x, mean, std


def denormalization(x, mean, std):
    B, D, L = x.shape
    if mean.shape[-1] == 1:
        x = x * (std[:, :D, 0].unsqueeze(-1).repeat(1, 1, L))
        x = x + (mean[:, :D, 0].unsqueeze(-1).repeat(1, 1, L))
    x = x * (std[:, :D, :L])
    x = x + mean[:, :D, :L]
    return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation, drop_rate=0.1, bias=False):
        super(FFN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias), activation,
            nn.Linear(d_ff, d_model, bias=bias), nn.Dropout(drop_rate),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Mean_Std_MLP(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, pred_len,
                 drop_rate=0.1, bias=False, layer=1):
        # 初始化尝试！单位矩阵开始
        super(Mean_Std_MLP, self).__init__()
        self.mean_proj = nn.Sequential(
            nn.Linear(seq_len, d_model, bias=bias), nn.Dropout(drop_rate))
        self.std_proj = nn.Sequential(
            nn.Linear(seq_len, d_model, bias=bias), nn.Dropout(drop_rate))

        self.mean_ffn = nn.Sequential(*[
            FFN(d_model, d_ff, nn.ReLU(), drop_rate, bias)
            for _ in range(layer)
        ])
        self.std_ffn = nn.Sequential(*[
            FFN(d_model, d_ff, nn.ReLU(), drop_rate, bias)
            for _ in range(layer)
        ])
        self.mean_pred = nn.Linear(d_model, pred_len, bias=bias)
        self.std_pred = nn.Linear(d_model, pred_len, bias=bias)

    def forward(self, mean, std, m_all=None, s_all=None):
        if m_all is None and s_all is None:
            m_all, s_all = mean.mean(dim=-1, keepdim=True), std.mean(dim=-1, keepdim=True)
        mean_r, std_r = mean - m_all, std - s_all
        mean_r, std_r = self.mean_proj(mean_r), self.std_proj(std_r)
        mean_r, std_r = self.mean_ffn(mean_r), self.std_ffn(std_r)
        mean_r, std_r = self.mean_pred(mean_r), self.std_pred(std_r)
        mean, std = mean_r + m_all, F.relu(std_r + s_all) + 1e-5
        return mean, std


class ConEmbedding(nn.Module):
    def __init__(
            self, args, c_in=None, group=None
    ):
        super(ConEmbedding, self).__init__()
        self.alpha, self.Asym = args.alpha, args.Asym
        self.num_p, self.dwt_len = args.num_p, None

        self.zero = args.zero
        wave = args.wavelet
        if wave == 'wo_wavelet':
            self.level = 0
        else:
            self.len = 2 if wave == 'sym3' else 8  # coif3
            self.level = 1
            self.iwt = DWT1DInverse(wave=wave, mode='zero')
            self.dwt = DWT1DForward(wave=wave, mode='zero', J=1)
        self.MLPs = nn.ModuleList(self.dwt_proj_init(args))

        self.d_model, self.d_ff = args.d_model, args.d_ff

        self.kernel = kernel = args.kernel
        self.pad = nn.ReplicationPad1d(padding=(kernel // 2, kernel // 2 - ((kernel + 1) % 2)))
        self.avg = nn.Sequential(
            nn.AvgPool1d(kernel_size=args.kernel, stride=1),
            nn.ReplicationPad1d(padding=(kernel // 2, kernel // 2 - ((kernel + 1) % 2)))
        )

        self.channel_pad = nn.ConstantPad1d((0, group - c_in % group), 0) if c_in % group > 0 else nn.Identity()
        self.latent_proj = nn.Sequential(
            nn.Linear(args.seq_len // args.num_p, args.d_model),
            nn.Dropout(p=args.dropout)
        )

    def forward(self, x, x_mark=None):
        x = torch.cat([x, x_mark], dim=-1).transpose(-1, 1) \
            if x_mark is not None else x.transpose(-1, 1)
        x, ms = self.dc_norm(x) if self.level > 0 else self.mov_norm(x)
        x = rearrange(x, 'B ch (num_p p) -> B ch num_p p', num_p=self.num_p)
        x = self.latent_proj(x)
        if x_mark is not None:
            x, x_mark = x[:, :-4, ...], x[:, -4:, ...]
        if self.Asym:
            x = self.channel_pad(x.transpose(1, -1)).transpose(1, -1)
        return x, x_mark, ms

    def dwt_proj_init(self, args):
        mlp_list = []
        d_model, d_ff = args.d_model, args.d_ff
        seq_len, pred_len = args.seq_len, args.pred_len
        if self.level > 0:
            for i in range(self.level):
                seq_len = seq_len // 2 + self.len
                pred_len = pred_len // 2 + self.len
                mlp_list.append(Mean_Std_MLP(seq_len, d_model, d_ff, pred_len,
                                             layer=args.e_layers - 2, bias=False, drop_rate=args.dropout))
            mlp_list.append(copy.deepcopy(mlp_list[-1]))
            self.dwt_len = seq_len
        else:
            mlp_list.append(Mean_Std_MLP(seq_len, d_model, d_ff, pred_len,
                                         layer=args.e_layers - 2, bias=False, drop_rate=args.dropout))
        return mlp_list

    def norm(self, x):
        # new_x = x.unfold(dimension=-1, size=self.kernel, step=1)
        # x = new_x.x(dim=-1)
        # deta = ((new_x - x.unsqueeze(-1)) ** 2).detach()
        # var = torch.sqrt(deta.x(dim=-1)) + 1e-5
        # x, var = self.pad(x), self.pad(var)
        # x = (x - x) / var

        mean = self.avg(x).detach()
        x = (x - mean).detach()
        std = self.avg(torch.abs(x)).detach() + 1e-5
        x = x / std
        return x, (mean, std)

    def denorm(self, x, ms, ck=None):
        (mc, mt), (sc, st) = torch.unbind(ms[0], dim=-2), torch.unbind(ms[1], dim=-2)
        xc, xt = torch.unbind(x, dim=-2)
        x, m, s = ck * xc + (1 - ck) * xt, ck * mc + (1 - ck) * mt, ck * sc + (1 - ck) * st
        x = denormalization(x, m, s)
        return x

    def dc_norm(self, x):
        ac, dc = self.dwt(x)
        ac_ = normalization(ac)[0]
        ac, (mean_ac, std_ac) = self.norm(ac)
        dc, (mean_dc, std_dc) = self.norm(dc[0])

        mean_ac, std_ac = self.MLPs[-1](mean_ac, std_ac)
        mean_dc, std_dc = self.MLPs[0](mean_dc, std_dc)

        freq = math.ceil(self.dwt_len / 2 * self.alpha) * 2
        dc, ac_ = torch.fft.fft(dc), torch.fft.fft(ac_)
        amp_fre = torch.argsort(torch.abs(ac_), descending=True)[..., :freq]
        amp_val = self.alpha * ac_.gather(-1, amp_fre) + (1 - self.alpha) * dc.gather(-1, amp_fre)
        dc = dc.scatter(-1, amp_fre, amp_val)
        dc = torch.fft.ifft(dc).real

        if self.zero is not None:
            if self.zero == 'ac':
                ac = torch.zeros_like(ac, device=ac.device)
            else:
                dc = torch.zeros_like(dc, device=dc.device)
        x, mean, std = self.iwt([ac, [dc]]), \
            self.iwt([mean_ac, [mean_dc]]), self.iwt([std_ac, [std_dc]])
        return x, (mean, std)

    def mov_norm(self, x):
        x, (mean, std) = self.norm(x)
        return x, self.MLPs[0](mean, std)


class ChannelEmbedding(nn.Module):
    def __init__(
            self, seq_len, d_model, c_in, group, dropout=0.1
    ):
        super(ChannelEmbedding, self).__init__()
        self.channel_pad = nn.ConstantPad1d((0, group - c_in % group), 0) if c_in % group > 0 else nn.Identity()
        self.latent_proj = nn.Sequential(nn.Linear(seq_len, d_model), nn.Dropout(p=dropout))

    def forward(self, x, x_mark=None, ck=0.3):
        # x = torch.cat([x, x_mark], dim=1)
        x = self.channel_pad(x.transpose(1, -1)).transpose(1, -1)
        return self.latent_proj(x)


def cosine_similarity(x):
    # x: (B, M, dim)
    norm = x.norm(p=2, dim=-1, keepdim=True)  # (B, M, 1)
    x_normalized = x / (norm + 1e-5)  # (B, M, dim)
    similarity = torch.matmul(x_normalized, x_normalized.transpose(-1, -2))  # (B, M, M)
    return similarity


class AggEmbedding(nn.Module):
    def __init__(
            self, args, c_in=None, num_g=None, group=None, k=4, num_p=None
    ):
        super(AggEmbedding, self).__init__()
        self.num_g = num_g
        self.group = group
        self.num_p = args.num_p if num_p is None else num_p
        self.d_model = args.d_model

        self.k = k
        self.pad = nn.ConstantPad1d((0, group - c_in % group), 0) \
            if c_in % group > 0 else nn.Identity()

        self.latent_proj = nn.Sequential(
            nn.Linear(args.seq_len // self.num_p, args.d_model),
            nn.Dropout(p=args.dropout)
        )

    def forward(self, x, x_mark=None):
        B, M, L = x.shape
        x = self.pad(x.transpose(-1, -2)).transpose(-1, -2)
        x = rearrange(x, 'B ch (num_p p) -> B ch num_p p', num_p=self.num_p)
        x = self.latent_proj(x)
        x = rearrange(x, 'B (num_g group) num_p p -> B num_g group num_p p', num_g=self.num_g)

        if x_mark is None:
            x_mark = torch.zeros((B, self.num_g, self.k, self.num_p, self.d_model)).to(x.device)
        else:
            x_mark = rearrange(x_mark, 'B time (num_p p) -> B time num_p p', num_p=self.num_p)
            x_mark = self.latent_proj(x_mark).unsqueeze(1).repeat(1, self.num_g, 1, 1, 1)
        x = torch.cat([x, x_mark], dim=2)
        return x


class Decompose(nn.Module):
    def __init__(self, args):
        super(Decompose, self).__init__()
        alpha = torch.full((1, args.enc_in), args.alpha)
        self.alpha = nn.Parameter(alpha)

    def forward(self, x):
        reg = copy.deepcopy(x)
        pre = x[..., 0]
        for i in range(1, reg.shape[-1]):
            pre = (1 - self.alpha) * pre + self.alpha * x[..., i]
            reg[..., i] = pre

        return reg


class decompose_emb(nn.Module):
    def __init__(self, args):
        super(decompose_emb, self).__init__()
        self.num_p = args.seq_len // args.period if args.num_p is None else args.num_p
        self.d_model = args.d_model
        self.period = args.period

        self.filter = Decompose(args)
        self.latent_proj = nn.Linear(args.seq_len // self.num_p, args.d_model, bias=False)

    def forward(self, x, x_mark):
        season, trend = self.decompose(x.transpose(-1, 1))

        season = rearrange(season, 'B ch (num_p p) -> B ch num_p p', num_p=self.num_p)
        season = self.latent_proj(season)

        return season, trend

    def decompose(self, x):
        trend = self.filter(x)
        season = x - trend
        return season, trend


class PatchEmbed(nn.Module):
    def __init__(self, args, num_p=1, d_model=None):
        super(PatchEmbed, self).__init__()
        self.num_p = num_p
        self.patch = args.seq_len // self.num_p
        self.d_model = args.d_model if d_model is None else d_model

        self.proj = nn.Sequential(
            nn.Linear(self.patch, self.d_model, False),
            nn.Dropout(args.dropout)
        )

    def forward(self, x, x_mark):
        x = torch.cat([x, x_mark], dim=-1).transpose(-1, -2)
        x = self.proj(x.reshape(*x.shape[:-1], self.num_p, self.patch))
        return x
