# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



# 一些基础定义和模块


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBiasLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ChannelTransposedSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        _, _, h, w = x.shape
        q, k, v = self.qkv2(self.qkv1(x)).chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, in_features, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class OverlappingPatchProjection(nn.Module):
    def __init__(self, in_channels=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)



# 共享特征编码器


def channel_range_prior(x):
    return torch.amax(x, dim=1, keepdim=True) - torch.amin(x, dim=1, keepdim=True)


class PriorGatedFeedForwardNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()

        expanded_dim = dim * 2
        mask_dim = dim // 2
        mask_branch_dim = dim // 4

        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, expanded_dim, 1),
            nn.GELU(),
        )

        self.conv_dw = nn.Sequential(
            nn.Conv2d(expanded_dim, expanded_dim, kernel_size=3, padding=1, groups=expanded_dim, padding_mode='reflect'),
            nn.GELU(),
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(expanded_dim, dim, 1),
            nn.GELU(),
        )
        self.mask_in = nn.Sequential(
            nn.Conv2d(1, mask_dim, 1),
            nn.GELU(),
        )
        self.mask_dw_conv_1 = nn.Sequential(
            nn.Conv2d(mask_branch_dim, 1, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Sigmoid(),
        )
        self.mask_dw_conv_2 = nn.Sequential(
            nn.Conv2d(mask_branch_dim, 1, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.Sigmoid(),
        )
        self.mask_out = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.GELU(),
        )

    def forward(self, x, mask):
        x1, x2 = self.conv_dw(self.conv_init(x)).chunk(2, dim=1)
        mask1, mask2 = self.mask_in(mask).chunk(2, dim=1)

        mask1 = self.mask_dw_conv_1(mask1)
        mask2 = self.mask_dw_conv_2(mask2)

        x = self.conv_fina(torch.cat((mask1 * x1, mask2 * x2), dim=1))
        mask = self.mask_out(torch.cat((mask1, mask2), dim=1))

        return x, mask


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        self.bn = nn.GroupNorm(8, out_channels * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        batch, _, h, w = x.shape
        ffted = torch.fft.rfft2(x.float(), norm='ortho')

        ffted = torch.stack((ffted.real, ffted.imag), dim=2)
        ffted = ffted.reshape(batch, -1, ffted.shape[-2], ffted.shape[-1]).to(x.dtype)
        ffted = self.gelu(self.bn(self.conv_layer(ffted))).float()

        ffted = ffted.reshape(batch, -1, 2, ffted.shape[-2], ffted.shape[-1])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)

        return torch.fft.irfft2(ffted, s=(h, w), norm='ortho').to(x.dtype)


class FourierResidualInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.FFC = FourierUnit(dim * 2, dim * 2)

    def forward(self, x):
        return self.FFC(x) + x


class FusedFourierConvolutionMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mixer_global = FourierResidualInteraction(dim)

        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim * 2, 1), nn.GELU())
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        self.ca_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU(),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        branch_3x3, branch_5x5 = self.conv_init(x).chunk(2, dim=1)
        local_3x3 = self.dw_conv_1(branch_3x3)
        local_5x5 = self.dw_conv_2(branch_5x5)

        global_feature = self.mixer_global(torch.cat((local_3x3, local_5x5), dim=1))
        out = self.ca_conv(global_feature)
        return self.ca(out) * out


class FARLModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)
        self.mixer = FusedFourierConvolutionMixer(dim)
        self.ffn = PriorGatedFeedForwardNetwork(dim)

    def forward(self, inputs):
        x, mask = inputs
        x = x + self.mixer(self.norm1(x))
        ffn_out, mask = self.ffn(self.norm2(x), mask)
        return x + ffn_out, mask


class HistoricalStateAttentionCascade(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim, 1, 1) * 0.02)
        self.norm = LayerNorm(dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, past_states, current_state):
        states = past_states + [current_state]
        keys = torch.stack([self.norm(state) for state in states], dim=1)
        values = torch.stack(states, dim=1)

        scores = (keys * self.query).sum(dim=2)
        scores = scores / (scores.std(dim=1, keepdim=True) + 1e-5)
        weights = F.softmax(scores * self.scale, dim=1)
        return (weights.unsqueeze(2) * values).sum(dim=1)


class SharedFeatureEncoder(nn.Module):
    def __init__(self, dim, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([FARLModule(dim) for _ in range(num_blocks)])
        self.hsac_layers = nn.ModuleList(
            [HistoricalStateAttentionCascade(dim) for _ in range(num_blocks - 1)]
        )

    def forward(self, x):
        mask = channel_range_prior(x)
        past_states = [x]

        x, mask = self.blocks[0]((x, mask))
        for hsac, block in zip(self.hsac_layers, self.blocks[1:]):
            aggregated = hsac(past_states, x)
            past_states.append(x)
            x, mask = block((aggregated, mask))

        return x



# 特征提取双分支


class FourierModulator(nn.Module):
    def __init__(self, dim, h=128, w=128):
        super().__init__()
        self.num_rows = h
        self.num_cols = w
        self.weights_real = nn.Parameter(torch.empty(1, dim, h, w // 2 + 1))
        self.weights_imag = nn.Parameter(torch.empty(1, dim, h, w // 2 + 1))
        nn.init.xavier_normal_(self.weights_real)
        nn.init.xavier_normal_(self.weights_imag)
        self.to_weight = nn.Sequential(nn.Conv2d(dim, dim, 1))

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        _, _, h, w = x.shape

        spectrum = torch.fft.rfft2(x, s=(h, w), dim=(-2, -1), norm='ortho')
        weight_real = self.weights_real
        weight_imag = self.weights_imag

        if h != self.num_rows or w != self.num_cols:
            target_size = (h, w // 2 + 1)
            weight_real = F.interpolate(weight_real, size=target_size, mode='bilinear', align_corners=False)
            weight_imag = F.interpolate(weight_imag, size=target_size, mode='bilinear', align_corners=False)

        out_real = spectrum.real * weight_real - spectrum.imag * weight_imag
        out_imag = spectrum.real * weight_imag + spectrum.imag * weight_real
        out = torch.fft.irfft2(
            torch.complex(out_real, out_imag),
            s=(h, w),
            dim=(-2, -1),
            norm='ortho',
        )
        return self.to_weight(out).to(orig_dtype)


class BaseFeatureExtractionBranch(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=1.0, qkv_bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = ChannelTransposedSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.fft_modulator = FourierModulator(dim, h=128, w=128)
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, ffn_expansion_factor=ffn_expansion_factor)
        self.fft_scale = nn.Parameter(torch.ones(1) * 0.15)

    def forward(self, x):
        spatial = self.attn(self.norm1(x))
        spectral = self.fft_modulator(x)
        x_fused = spatial + self.fft_scale * spectral
        x = x + x_fused
        return x + self.mlp(self.norm2(x))


class AdaptiveAnalysisSynthesisFrequencyDecomposition(nn.Module):
    """Learnable analysis/synthesis with one shared depth-wise filter bank."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.empty(in_channels * 4, 1, 4, 4))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, rev=False):
        n, _, h, w = x.shape

        if not rev:
            out = F.conv2d(x, self.weight, stride=2, padding=1, groups=self.in_channels)
            out = out.reshape(n, self.in_channels, 4, h // 2, w // 2)
            return out.transpose(1, 2).reshape(n, self.in_channels * 4, h // 2, w // 2)

        out = x.reshape(n, 4, self.in_channels, h, w)
        out = out.transpose(1, 2).reshape(n, self.in_channels * 4, h, w)
        return F.conv_transpose2d(out, self.weight, stride=2, padding=1, groups=self.in_channels)


class DetailNode(nn.Module):
    """One AASFD + asymmetric affine coupling + synthesis residual node."""

    def __init__(self, channels=64):
        super().__init__()
        low_channels = channels
        high_channels = channels * 3

        self.irb_phi = InvertedResidualBlock(low_channels, high_channels, expand_ratio=2)
        self.irb_rho = InvertedResidualBlock(high_channels, low_channels, expand_ratio=2)
        self.irb_eta = InvertedResidualBlock(high_channels, low_channels, expand_ratio=2)
        self.aasfd = AdaptiveAnalysisSynthesisFrequencyDecomposition(channels)

    def forward(self, x):
        identity = x
        frequency_groups = self.aasfd(x)
        channels = x.shape[1]

        u_low = frequency_groups[:, :channels]
        u_high = frequency_groups[:, channels:]

        u_high = u_high + self.irb_phi(u_low)
        scale = torch.exp(torch.clamp(self.irb_rho(u_high), min=-2.0, max=2.0))
        u_low = u_low * scale + self.irb_eta(u_high)

        coupled_groups = torch.cat((u_low, u_high), dim=1)
        return self.aasfd(coupled_groups, rev=True) + identity


class DetailFeatureExtractionBranch(nn.Module):
    def __init__(self, dim=64, num_layers=3):
        super().__init__()
        self.net = nn.Sequential(*[DetailNode(channels=dim) for _ in range(num_layers)])
        self.output_norm = nn.GroupNorm(8, dim)
        self.output_act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return self.output_act(self.output_norm(x))



# 特征融合模块


class FrequencyAwareBaseFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_fuse = nn.Conv2d(dim * 2, dim, 1)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.ReLU(inplace=True),
        )

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, base_ir, base_vi):
        orig_dtype = base_ir.dtype
        base_ir = base_ir.float()
        base_vi = base_vi.float()

        spatial = self.spatial_fuse(torch.cat((base_ir, base_vi), dim=1))

        fft_ir = torch.fft.rfft2(base_ir, norm='ortho')
        fft_vi = torch.fft.rfft2(base_vi, norm='ortho')
        mag_ir = torch.abs(fft_ir)
        mag_vi = torch.abs(fft_vi)

        selector = mag_ir > mag_vi
        magnitude = torch.maximum(mag_ir, mag_vi)
        phase = torch.where(selector, torch.angle(fft_ir), torch.angle(fft_vi))
        spectral = torch.fft.irfft2(
            magnitude * torch.exp(1j * phase),
            s=base_ir.shape[-2:],
            norm='ortho',
        )

        fused = self.final_fuse(torch.cat((spatial, spectral), dim=1))
        return (fused + base_ir + base_vi).to(orig_dtype)


class SpatialGradient(nn.Module):
    def __init__(self, dim):
        super().__init__()
        kernel_v = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_h = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer('weight_v', kernel_v.repeat(dim, 1, 1, 1))
        self.register_buffer('weight_h', kernel_h.repeat(dim, 1, 1, 1))
        self.dim = dim

    def forward(self, x):
        grad_v = F.conv2d(x, self.weight_v, padding=1, groups=self.dim)
        grad_h = F.conv2d(x, self.weight_h, padding=1, groups=self.dim)
        return torch.sqrt(grad_v.square() + grad_h.square() + 1e-6)


class GradientGatedDetailFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gradient = SpatialGradient(dim)
        self.channel_mixer = nn.Conv2d(dim, dim, 1, bias=True)
        self.gate_scale = nn.Parameter(torch.ones(1) * 5.0)
        self.gate_bias = nn.Parameter(torch.zeros(1))

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, detail_ir, detail_vi):
        orig_dtype = detail_ir.dtype
        detail_ir = detail_ir.float()
        detail_vi = detail_vi.float()

        grad_ir = self.gradient(detail_ir)
        grad_vi = self.gradient(detail_vi)
        grad_ir = grad_ir / (grad_ir.mean(dim=(-2, -1), keepdim=True) + 1e-6)
        grad_vi = grad_vi / (grad_vi.mean(dim=(-2, -1), keepdim=True) + 1e-6)

        gate_ir = torch.sigmoid(self.gate_scale * (grad_ir - self.gate_bias))
        gate_vi = torch.sigmoid(self.gate_scale * (grad_vi - self.gate_bias))

        fused = gate_ir * detail_ir + gate_vi * detail_vi
        return self.channel_mixer(fused).to(orig_dtype)



# Encoder和Decoder


class FrequencyAwareEncoder(nn.Module):
    def __init__(self, inp_channels=1, dim=64, num_farl_blocks=4, num_heads=8):
        super().__init__()
        self.patch_projection = OverlappingPatchProjection(inp_channels, dim)
        self.shared_encoder = SharedFeatureEncoder(dim=dim, num_blocks=num_farl_blocks)
        self.base_branch = BaseFeatureExtractionBranch(dim=dim, num_heads=num_heads)
        self.detail_branch = DetailFeatureExtractionBranch(dim=dim)
        self.input_proj = nn.Conv2d(inp_channels, dim, kernel_size=1)

    def forward(self, inp_img):
        common_feature = self.shared_encoder(self.patch_projection(inp_img))
        base_feature = self.base_branch(common_feature) + self.input_proj(inp_img)
        detail_feature = self.detail_branch(common_feature)
        return base_feature, detail_feature, common_feature


class ReconstructionDecoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        dim=64,
        num_blocks=4,
        ffn_expansion_factor=2,
        bias=False,
    ):
        super().__init__()
        self.w_base = nn.Parameter(torch.ones(1))
        self.w_detail = nn.Parameter(torch.ones(1))
        self.reduce_channel = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        decoder_blocks = []
        for _ in range(num_blocks):
            decoder_blocks.extend(
                [
                    nn.GroupNorm(8, dim),
                    Mlp(in_features=dim, ffn_expansion_factor=ffn_expansion_factor),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.body = nn.Sequential(*decoder_blocks)

        self.output = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    def forward(self, reference_image, base_feature, detail_feature):
        x = torch.cat(
            (base_feature * self.w_base, detail_feature * self.w_detail),
            dim=1,
        )
        decoder_input = self.reduce_channel(x)

        x = decoder_input
        for i in range(0, len(self.body), 3):
            norm, mlp, act = self.body[i], self.body[i + 1], self.body[i + 2]
            x = act(x + mlp(norm(x)))

        out = self.output(x)
        if reference_image is not None:
            out = out + reference_image
        return torch.sigmoid(out), decoder_input
