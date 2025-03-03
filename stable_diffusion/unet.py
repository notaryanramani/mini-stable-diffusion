import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention, CrossAttention
from .utils import Config
from typing import List, Tuple


config = Config()


class UNetResidual(nn.Module):
    def __init__(self, channels:int, n_time:int = config.TIME_DIM) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(config.BATCH_SIZE, channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.time_ln = nn.Linear(n_time, channels)

        self.gn_merge = nn.GroupNorm(config.BATCH_SIZE, channels)
        self.conv_merge = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        fx = self.gn(x)
        fx = F.silu(fx)
        fx = self.conv(fx)

        time = self.time_ln(time)

        mx = fx + time.unsqueeze(-1).unsqueeze(-1)
        mx = self.gn_merge(mx)
        mx = F.silu(mx)
        mx = self.conv_merge(mx)

        out = mx + x
        return out


class UNetAttention(nn.Module):
    def __init__(self, n_head:int, n_embd:int, d_prompt:int = config.PROMPT_DIM) -> None:
        super().__init__()
        channels = n_head * n_embd

        self.gn = nn.GroupNorm(config.BATCH_SIZE, channels)
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.ln1 = nn.LayerNorm(channels)
        self.att = SelfAttention(n_head, channels)
        self.ln2 = nn.LayerNorm(channels)
        self.cross_att = CrossAttention(n_head, channels, d_prompt)
        self.ln3 = nn.LayerNorm(channels)

        self.gelu_linear = nn.Linear(channels, channels * 4 * 2)
        self.gelu_linear2 = nn.Linear(channels * 4, channels)

        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)


    def forward(self, x:torch.Tensor, prompt:torch.Tensor) -> torch.Tensor:
        fx = self.gn(x)
        fx = self.conv_in(fx)

        B, C, H, W  = x.shape
        fx = fx.view(B, C, H*W).transpose(-2, -1)

        ax = fx + self.att(self.ln1(fx))
        cx = ax + self.cross_att(self.ln2(ax), prompt)

        sx = cx.clone()
        cx = self.ln3(cx)
        gx, gate = self.gelu_linear(cx).chunk(2, dim = -1)
        gx = gx * F.gelu(gate)
        gx = self.gelu_linear2(gx)
        gx = gx + sx

        gx = gx.transpose(-2, -1).reshape(B, C, H, W)
        out = x + self.conv_out(gx)

        return out


class UNetEncoder(nn.Module):
    def __init__(self, channels:int=config.UNET_INIT_CHANNELS) -> None:
        super().__init__()

        # (B, 4, H/8, W/8) -> (B, channels, H/8, W/8)
        self.block1 = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1),
            nn.GroupNorm(config.BATCH_SIZE, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.res1 = UNetResidual(channels)
        self.att1 = UNetAttention(8, channels // 8)

        # (B, channels, H / 8, W / 8) -> (B, channels * 2, H / 16, W / 16)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(config.BATCH_SIZE, channels * 2),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.res2 = UNetResidual(channels * 2)
        self.att2 = UNetAttention(8, channels * 2 // 8)

        # (B, channels * 2, H/16, W/32) -> (B, channels * 4, H/32, W/32)
        self.block3 =  nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(config.BATCH_SIZE, channels * 4),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.res3 = UNetResidual(channels * 4)
        self.att3 = UNetAttention(8, channels * 4 // 8)

        self.blocks = nn.ModuleList([
            self.block1,
            self.block2,
            self.block3,
        ])

        self.res = nn.ModuleList([
            self.res1,
            self.res2,
            self.res3,
        ])

        self.att = nn.ModuleList([
            self.att1,
            self.att2,
            self.att3,
        ])

    def forward(self, x:torch.Tensor, prompt:torch.Tensor, time:torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for block, res, att in zip(self.blocks, self.res, self.att):
            x = block(x)
            x = res(x, time)
            x = att(x, prompt)
            skips.append(x)
        return x, skips


class UNetDecoder(nn.Module):
    def __init__(self, channels:int = config.UNET_INIT_CHANNELS) -> None:
        super().__init__()
        # (B, channels * 4, H/32, W/32) --[concat]-> (B, channels * 8, H/32, W/32) -> (B, channels * 2, H/16, W/16)
        self.block1 = nn.Sequential(
            nn.Conv2d(channels * 8, channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(config.BATCH_SIZE, channels * 4),
            nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.SiLU(),
        )

        self.res1 = UNetResidual(channels * 2)
        self.att1 = UNetAttention(8, channels * 2 // 8)


        # (B, channels * 2, H/16, W/16) --[concat]-> (B, channels * 4, H/16, W/16) -> (B, channels, H/8, W/8)
        self.block2 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(config.BATCH_SIZE, channels * 2),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.SiLU(),
        )

        self.res2 = UNetResidual(channels)
        self.att2 = UNetAttention(8, channels // 8)

        # (B, channels, H/8, W/8) --[concat]-> (B, channels * 2, H/8, W/8) -> (B, channels, H/8, W/8)
        self.block3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.res3 = UNetResidual(channels)
        self.att3 = UNetAttention(8, channels // 8)

        # (B, channels, H/8, W/8) -> (B, 4, H/8, H/8)
        self.output_layer = nn.Sequential(
            nn.GroupNorm(config.BATCH_SIZE, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 4, kernel_size=3, padding=1),
        )

        self.blocks = nn.ModuleList([
            self.block1,
            self.block2,
            self.block3,
        ])

        self.res = nn.ModuleList([
            self.res1,
            self.res2,
            self.res3,
        ])

        self.att = nn.ModuleList([
            self.att1,
            self.att2,
            self.att3,
        ])

    def forward(self, x:torch.Tensor, skips:List[torch.Tensor], prompt:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        for block, res, att in zip(self.blocks, self.res, self.att):
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
            x = res(x, time)
            x = att(x, prompt)
        x = self.output_layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, channels:int=config.UNET_INIT_CHANNELS) ->  None:
        super().__init__()
        # (B, 4, H/8, W/8) -> (B, channels * 8, H/64, H/64)
        self.encoders = UNetEncoder(channels)

        # (B, channels * 8, H/64, W/64) -> (B, channels * 16, H/128, W/128)
        self.bottleneck_in = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=3, padding=1)
        )

        self.res = UNetResidual(channels * 8)
        self.att = UNetAttention(4, channels * 8 // 4)

        #  (B, channels * 16, H/128, W/128) -> (B, channels * 8, H/64, W/64)
        self.bottleneck_out = nn.Sequential(
            nn.Conv2d(channels * 8, channels * 4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
        )

        # (B, channels * 8, H/64, W/64) -> (B, 4, H/8, W/8)
        self.decoders = UNetDecoder(channels)

    def forward(self, x:torch.Tensor, prompt:torch.Tensor, time:torch.Tensor) ->  torch.Tensor:
        x, skips = self.encoders(x, prompt, time)

        x = self.bottleneck_in(x)
        x = self.res(x, time)
        x = self.att(x, prompt)
        x = self.bottleneck_out(x)

        x = self.decoders(x, skips, prompt, time)
        return x
