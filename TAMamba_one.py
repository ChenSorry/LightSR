import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable

from einops import rearrange, repeat
from basicsr.archs.arch_util import flow_warp
from basicsr.utils.registry import ARCH_REGISTRY
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr:  # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                nn.Conv2d(num_feat // compress_ratio, num_feat // compress_ratio, 3, 1, 1,
                          groups=num_feat // compress_ratio),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                nn.Conv2d(num_feat, num_feat, 3, 1, padding=2, groups=num_feat, dilation=2),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            change_direction=1,
            image_size=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.change_direction = change_direction

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        # self.relative_position_bias_table = RelativePositionEmbedding(image_size, d_model)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def sass_hor(self, hw_shape):
        H, W = hw_shape
        L = H * W
        o1, o2 = [], []
        d1, d2 = [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]

        i, j = 0, 0
        j_d = 'right'
        while i < H:
            assert j_d in ['right', 'left']
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == 'right':
                if j < W - 1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i + 1
                    d1.append(3)
                    j_d = 'left'
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                    j_d = 'left'
                else:
                    i = i + 1
                    d1.append(3)
                    j_d = 'right'
        d1 = [0] + d1[:-1]

        if H % 2 == 1:
                i, j = H - 1, W - 1
                j_d = 'left'
        else:
            i, j = H - 1, 0
            j_d = 'right'

        while i > -1:
            assert j_d in ['left', 'right']
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if j_d == 'right':
                if j < W - 1:
                    j = j + 1
                    d2.append(1)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = 'left'
            else:
                if j > 0:
                        j = j -1
                        d2.append(2)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = 'right'
        d2 = [0] + d2[:-1]

        return (tuple(o1), tuple(o2)), (tuple(o1_inverse), tuple(o2_inverse)), (tuple(d1), tuple(d2))

    def sass_ver(self, hw_shape):
        H, W = hw_shape
        L = H * W
        o1, o2 = [], []
        d1, d2 = [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        i_d = 'down'
        while j < W:
            assert i_d in ['up', 'down']
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if i_d == 'down':
                if i < H - 1:
                    i = i + 1
                    d1.append(4)
                else:
                    j = j + 1
                    d1.append(1)
                    i_d = 'up'
            else:
                if i > 0:
                    i = i - 1
                    d1.append(3)
                else:
                    j = j + 1
                    d1.append(1)
                    i_d = 'down'
        d1 = [0] + d1[:-1]

        if H % 2 == 1:
            i, j = H - 1, W - 1
            i_d = 'up'
        else:
            i, j = 0, W - 1
            i_d = 'down'

        while j > 0:
            assert i_d in ['up', 'down']
            idx = i * W + j
            o2_inverse[idx] = len(o1)
            o2.append(idx)
            if i_d == 'down':
                if i < H - 1:
                    i = i + 1
                    d2.append(4)
                else:
                    j = j - 1
                    d2.append(1)
                    i_d = 'up'
            else:
                if i > 0:
                    i = i - 1
                    d2.append(3)
                else:
                    j = j - 1
                    d2.append(1)
        d2 = [0] + d2[:-1]

        return (tuple(o1), tuple(o2)), (tuple(o1_inverse), tuple(o2_inverse)), (tuple(d1), tuple(d2))

    def forward_hor(self, x):
        B, C, H, W = x.shape
        L = H * W
        hw_shape = (H, W)
        order, inverse_order, directions = self.sass_hor(hw_shape=hw_shape)
        K = 4
        x_hw = x.view(B, -1, L).view(B, 1, -1, L)
        x_hw_inverse = torch.flip(x_hw, dims=[-1]).view(B, 1, -1, L)
        x_hwwh = torch.cat([x_hw, x_hw_inverse], dim=1)
        x_hw_pa = x_hw[:, :, order[0]].view(B, 1, -1, L)
        x_hw_pa_inver = x_hw[:, :, order[1]].view(B, 1, -1, L)
        xs = torch.cat([x_hwwh, x_hw_pa, x_hw_pa_inver], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = x_hwwh.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Bs = Bs[:, 2] + directions[0]
        Bs = Bs[:, 3] + directions[1]
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        y1 = out_y[:, 0]
        y2 = torch.flip(out_y[:, 1], dims=[-1])
        y3 = out_y[:, 2][:, :, :, inverse_order[0]]
        y4 = out_y[:, 3][:, :, :, inverse_order[1]]

        return y1, y2, y3, y4

    def forward_ver(self, x):
        B, C, H, W = x.shape
        L = H * W
        hw_shape = (H, W)
        order, inverse_order, directions = self.sass_ver(hw_shape=hw_shape)
        K = 4
        x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        x_whhw = torch.cat([x_wh, torch.flip(x_wh, dims=[-1])], dim=1)
        x_wh_par = x.view(B, -1, L)[:, :, order[0]]
        x_hw_par = x.view(B, -1, L)[:, :, order[1]]
        xs = torch.cat([x_whhw, x_wh_par, x_hw_par], dim=1).view(B, K, -1, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Bs[:, 2, :, :] += directions[0]
        Bs[:, 3, :, :] += directions[1]
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        y1 = torch.transpose(out_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y2 = torch.transpose(torch.flip(out_y[:, 1], dims=[-1]).view(B, -1, W, H), dim0=2, dim1=3).view(B, -1, L)
        y3 = out_y[:, 2][:, :, :, inverse_order[0]].view(B, -1, L)
        y4 = out_y[:, 3][:, :, :, inverse_order[1]].view(B, -1, L)
        return y1, y2, y3, y4

    # def forward_core(self, x: torch.Tensor):
    #     B, C, H, W = x.shape
    #     L = H * W
    #     K = 4
    #     x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
    #                          dim=1).view(B, 2, -1, L)
    #     xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
    #
    #     x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
    #     dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    #     dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
    #     xs = xs.float().view(B, -1, L)
    #     dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
    #     Bs = Bs.float().view(B, K, -1, L)
    #     Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
    #     Ds = self.Ds.float().view(-1)
    #     As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
    #     dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
    #     out_y = self.selective_scan(
    #         xs, dts,
    #         As, Bs, Cs, Ds, z=None,
    #         delta_bias=dt_projs_bias,
    #         delta_softplus=True,
    #         return_last_state=False,
    #     ).view(B, K, -1, L)
    #     assert out_y.dtype == torch.float
    #
    #     inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
    #     wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    #     invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    #
    #     return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        if self.change_direction == 0:
            y1, y2, y3, y4 = self.forward_hor(x)
        else:
            y1, y2, y3, y4 = self.forward_ver(x)

        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            change_direction=0,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate,
                                   change_direction=change_direction, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class TAC(nn.Module):
    def __init__(self, n_feats, window_size, init_thresh=0.5, sharpness=10.0):
        super().__init__()
        self.window_size = window_size
        self.sharpness = sharpness

        self.in_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 4, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.thresh = nn.Parameter(torch.tensor(init_thresh))

    def forward(self, x):
        x = self.in_conv(x)

        x = torch.mean(x, keepdim=True, dim=1)

        x = rearrange(x, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        var_x = torch.var(x, dim=-1) # (B, N)

        soft_mask = 1.0 - torch.sigmoid(self.sharpness * (self.thresh - var_x))

        binary_mask = (soft_mask > 0.5).float()

        mask = binary_mask.detach() - soft_mask.detach() + soft_mask

        return mask


class TAG(nn.Module):
    def __init__(self, n_feats, window_size, bias=True, is_deformable=True):
        super().__init__()
        self.dim = n_feats
        self.window_size = window_size

        k = 3
        d = 2

        self.project_v = nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.project_q = nn.Linear(n_feats, n_feats, bias=bias)
        self.project_k = nn.Linear(n_feats, n_feats, bias=bias)

        # Conv
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=k, padding=k//2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, kernel_size=k, padding=((k//2)*d), groups=n_feats, dilation=d)
        )
        self.prject_out = nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0, bias=bias)

        self.act = nn.GELU

        self.mask_pre = TAC(n_feats=n_feats, window_size=window_size)

    def forward(self, x):
        B, C, H, W = x.shape

        v = self.project_v(x)

        mask = self.mask_pre(v)

        q = x
        k = x
        qk = torch.cat([q, k], dim=1)

        v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        N = v.shape[0]
        v1 = v * mask
        v2 = v * (1-mask)
        qk = qk * mask

        v1 = rearrange(v1, 'b n (dh dw c) -> (b n) (dh dw) c', n=N, dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk, 'b n (dh dw c) -> b (n dh dw) c', n=N, dh=self.window_size, dw=self.window_size)

        q, k = torch.chunk(qk, 2, dim=2)
        q = self.project_q(q)
        k = self.project_k(k)

        q = rearrange(q, 'b (n dh dw) c -> (b n) (dh dw) c', n=N, dh=self.window_size, dw=self.window_size)
        k = rearrange(k, 'b (n dh dw) c -> (b n) (dh dw) c', n=N, dh=self.window_size, dw=self.window_size)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn @ v1

        f_attn = rearrange(f_attn, '(b n) (dh dw) c -> b n (dh dw c)', b=B, n=N, dh=self.window_size, dw=self.window_size)

        v2 = self.conv_spatial(v2)

        f_attn = rearrange(f_attn, 'b n (dh dw c) -> b c (h dh) (w dw)', h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size, c=C)

        attn_out = f_attn + v2

        out = self.project_out(attn_out)

        return out


class TA_trans(nn.Module):
    def __init__(self, n_feats, window_size):
        super().__init__()
        self.n_feats = n_feats
        self.norm1 = nn.LayerNorm(n_feats)
        self.TAG = TAG(n_feats, window_size)
        self.norm2 = nn.LayerNorm(n_feats)
        self.ffn = GatedFeedForward(n_feats)

    def forward(self, x):
        b, c, h, w = x.shape
        res = self.TAG(x)
        x = x + res

        x = x .view(b, c, -1).transpose(-2, -1).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)

        res = self.ffn(x)
        x = x + res

        x = x.view(b, c, -1).transpose(-2, -1).contiguous()
        x = self.norm2(x)
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        return x


class GatedFeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.):
        super().__init__()
        self.dim = dim

        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Residual_block(nn.Module):
    def __init__(self,
                 n_feats,
                 window_size,
                 hidden_dim: int = 0,
                 drop_path=None,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 expand: float = 2.,
                 depth=0,
                 is_light_sr: bool = False,
                 **kwargs):
        super().__init__()

        self.TA_trans = TA_trans(n_feats=n_feats,
                                 window_size=window_size)


        self.VSSBlock = nn.ModuleList([
            VSSBlock(hidden_dim=hidden_dim,
                     drop_path=drop_path[i],
                     norm_layer=norm_layer,
                     attn_drop_rate=attn_drop_rate,
                     d_state=d_state,
                     expand=expand,
                     change_direction=i // 2) for i in range(depth)
        ])

        # self.VSSBlock = VSSBlock(hidden_dim=hidden_dim,
        #                          drop_path=drop_path,
        #                          norm_layer=norm_layer,
        #                          attn_drop_rate=attn_drop_rate,
        #                          d_state=d_state,
        #                          expand=expand)
        # self.conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        shortcut = x
        x_trans = self.TA_trans(x)
        x_ssm = self.VSSBlock(x_trans)
        out = self.conv(x_ssm) + shortcut

        return out


class TAMamba(nn.Module):
    def __init__(self,
                 n_colors=3,
                 n_feats=60,
                 window_size=16,
                 scale=4,
                 depths=(3, 3, 3, 3),
                 drop_path_rate=0.1,
                 d_state=16,
                 mlp_ratio=2.):
        super().__init__()
        self.n_feats = n_feats
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.num_layer = len(depths)

        self.head = nn.Conv2d(n_colors, n_feats, kernel_size=3, stride=1, padding=1)

        self.scale = scale

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.body = nn.ModuleList()
        for i_layer in range(self.num_layer):
            block = Residual_block(
                n_feats=n_feats,
                window_size=self.window_sizes,
                hidden_dim=n_feats,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio
            )
        # for i_depth in range(depths):
        #     block = Residual_block(
        #         n_feats=n_feats,
        #         window_size=self.window_sizes,
        #         hidden_dim=n_feats,
        #         drop_path=dpr[i_depth+1],
        #         norm_layer=nn.LayerNorm,
        #         attn_drop_rate=0,
        #         d_state=d_state,
        #         expand=self.mlp_ratio
        #
        #     )
            self.body.append(block)

        self.body_tail = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale)
        )

    def check_img_size(self, x):
        _, _, h, w = x.size()
        w_size = self.window_size

        mod_pad_h = (w_size - h % w_size) % w_size
        mod_pad_w = (w_size - w % w_size) % w_size

        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_img_size(x)

        x = self.head(x)
        res = x

        for blk in self.body:
            x = blk(x)

        x = self.body_tail(x)
        x = x + res
        x = self.tail(x)

        return x[:, :, 0:H*self.scale, 0:W*self.scale]


if __name__ == '__main__':

    x = torch.randn((1, 3, 32, 32))
    net = TAMamba(scale=4, n_feats=60)
    y = net(x)