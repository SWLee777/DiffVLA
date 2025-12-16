import torch
import torch.nn as nn
import math


class LFR(nn.Module):
    def __init__(self, clip_model):
        super(LFR, self).__init__()
        self.clip_model = clip_model

        self.target_size = max(self.clip_model.token_size)
        token_sizes = self.clip_model.token_size
        token_channels = self.clip_model.token_c

        self.res_proj = nn.ModuleList() 

        for i, (size, in_channels) in enumerate(zip(token_sizes, token_channels)):
            upsample_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=self.target_size / size, mode='bilinear', align_corners=False),
                nn.GroupNorm(1, in_channels),
                nn.Dropout2d(p=0.1)
            )
            self.add_module(f"{i}_upsample", upsample_block)

            self.res_proj.append(nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False))

    @torch.no_grad()
    def forward(self, tokens):
        align_features = []

        for i, token in enumerate(tokens):
            if token.ndim == 3:
                B, N, C = token.shape
                token = token[:, 1:, :] 
                spatial_size = int(math.sqrt(N - 1))
                token = token.view(B, spatial_size, spatial_size, C).permute(0, 3, 1, 2)

            upsample_block = getattr(self, f"{i}_upsample")
            token_residual = self.res_proj[i](token)
            aligned = upsample_block(token)

            aligned = aligned + token_residual

            align_features.append(aligned)

        return align_features