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


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

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
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                nn.Conv2d(num_feat//compress_ratio, num_feat // compress_ratio, 3, 1, 1,groups=num_feat//compress_ratio),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                nn.Conv2d(num_feat, num_feat, 3,1,padding=2,groups=num_feat,dilation=2),
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

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
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
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim,is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class MaskPre(nn.Module):
    def __init__(self, dim, window_size=8):
        super(MaskPre,self).__init__()

        self.window_size = window_size
        cdim = dim
        embed_dim = window_size ** 2 * dim

        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim // 4, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(cdim, cdim // 4, 1),
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim // 4, dim, 1),
            nn.Sigmoid(),
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim // 4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_x):

        x = self.in_conv(input_x)
        x_ = self.conv(input_x)

        ca = self.out_CA(x_)
        sa = self.out_SA(x_)

        x = torch.mean(x, keepdim=True, dim=1)

        x = rearrange(x, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        var_x = torch.var(x, keepdim=True, dim=-1)
        k_top = int(0.5 * N)
        _, indices = torch.topk(var_x.squeeze(-1), k_top, largest=False, sorted=False)

        mask = torch.ones_like(var_x)
        for batch_idx in range(var_x.size(0)):
            for window_idx in range(var_x.size(1)):
                if window_idx in indices[batch_idx]:
                    mask[batch_idx, window_idx, 0] = 0

        mask = self.out_mask(mask)

        return mask, ca, sa


class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable

        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.project_q = nn.Linear(dim, dim, bias=bias)
        self.project_k = nn.Linear(dim, dim, bias=bias)

        # Conv
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k // 2) * d), groups=dim, dilation=d))
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)

        self.act = nn.GELU()
        # Predictor
        self.route = MaskPre(dim, window_size)

    def forward(self, x, condition_global=None, mask=None, train_mode=False):
        N, C, H, W = x.shape

        v = self.project_v(x)

        mask, ca, sa = self.route(v)

        q = x
        k = x
        qk = torch.cat([q, k], dim=1)

        vs = v * sa

        v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        N_ = v.shape[1]
        v1, v2 = v * mask, vs * (1 - mask)
        qk1 = qk * mask

        v1 = rearrange(v1, 'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1, 'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1, k1 = torch.chunk(qk1, 2, dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn @ v1

        f_attn = rearrange(f_attn, '(b n) (dh dw) c -> b n (dh dw c)',
                           b=N, n=N_, dh=self.window_size, dw=self.window_size)

        attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)',
            h=H // self.window_size, w=W // self.window_size, dh=self.window_size, dw=self.window_size
        )

        out = attn_out
        out = self.act(self.conv_sptial(out)) * ca + out
        out = self.project_out(out)

        return out


class GatedFeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()
        self.dim = dim

        self.project_in = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
    def calculate_flops(self,x):
        _,_,H,W = x.shape
        flops = np.longlong(0)
        flops += H*W*self.dim*2*self.dim*2
        flops += H*W*2*self.dim*9*2
        flops += H*W*self.dim*self.dim*2
        return flops


class Block(nn.Module):
    def __init__(self, n_feats, window_size=8):
        super(Block,self).__init__()

        self.n_feats = n_feats
        self.norm1 = LayerNorm(n_feats)
        self.mixer = CAMixer(n_feats,window_size=window_size)
        self.norm2 = LayerNorm(n_feats)
        self.ffn = GatedFeedForward(n_feats)

    def forward(self, x):
        res = self.mixer(x)
        x = self.norm1(x + res)
        res = self.ffn(x)
        x = self.norm2(x+res)

        return x


class Residual_block(nn.Module):
    def __init__(self,
                 n_feats,
                 window_size,
                 hidden_dim: int = 0,
                 drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 expand: float = 2.,
                 is_light_sr: bool = False,
                 **kwargs,):
        super(Residual_block).__init__()

        self.Block = Block(n_feats=n_feats,
                           window_size=window_size)

        self.VSSBlock = VSSBlock(hidden_dim=hidden_dim,
                                 drop_path=drop_path,
                                 norm_layer=norm_layer,
                                 attn_drop_rate=attn_drop_rate,
                                 d_state=d_state,
                                 expand=expand)

        self.conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)

    def forward(self, x, x_size):
        short_cut = x
        x_mixer = self.Block(x)
        x_ssm = self.VSSBlock(x_mixer, x_size)
        out = self.conv(x_ssm) + short_cut

        return out


@ARCH_REGISTRY.register()
class MambaIR(nn.Module):
    def __init__(self, n_colors=3, n_feats=60, scale=4, title=None, depths=6, drop_path_rate=0.1, d_state=16, mlp_ratio=2.):
        super().__init__()
        self.n_feats = n_feats
        self.window_sizes = 16
        self.title = title
        self.mlp_ratio = mlp_ratio

        self.head = nn.Conv2d(n_colors, n_feats, kernel_size=3, stride=1, padding=1)

        self.scale = scale
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.body = nn.ModuleList()
        for i_depth in range(depths):
            block = Residual_block(
                n_feats=n_feats,
                window_size=self.window_sizes,
                hidden_dim=n_feats,
                drop_path=dpr[i_depth],
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio

            )
            self.body.append(block)

        self.body_tail = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors*(scale**2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale)
        )

    def check_img_size(self, x):
        _, _, h, w = x.size()
        w_size = self.window_sizes

        mod_pad_h = (w_size - h % w_size) % w_size
        mod_pad_w = (w_size - w % w_size) % w_size

        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_img_size(x)
        _, _, h, w = x.size()
        x_size = (h, w)

        x = self.head(x, x_size)
        short_cut = x.clone

        for blk in self.body:
            x = blk(x, x_size)

        x = self.body_tail(x)
        x = x + short_cut
        x = self.tail(x)

        return x[:, :, 0:H*self.scale, 0:W*self.scale]


if __name__ == '__main__':

    x = torch.randn((1,3,32,32))
    net = MambaIR(scale=4, n_feats=60)
    y = net(x)
    net.eval()
    num_parameters = sum(map(lambda x: x.numel(), net.parameters()))
    print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))
    print('{:>16s} : {:<.4f} [G]'.format('#FLOPs', net.flops(x)/10**9))
    print(net(x).shape)
