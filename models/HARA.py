import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class hara(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 2,
        groups: Optional[int] = None,
        dropout_p: float = 0.1
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        hidden = max(1, out_channels // reduction)
        g_out = groups if groups is not None else max(1, min(32, out_channels // 4))
        g_hid = max(1, min(32, hidden // 4))

        self.block = nn.Sequential(
            nn.Conv2d(out_channels, hidden, kernel_size=1, bias=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=g_hid, num_channels=hidden),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=g_out, num_channels=out_channels),
        )

        self.alpha = nn.Parameter(torch.zeros(1))

        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')
        for m in self.block:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        z = self.proj(x)                  # [B, C_out, H, W]
        out = z + self.alpha * self.block(z)
        return out


class HARA(nn.Module):
    def __init__(
        self,
        clip_model,
        target: int,
        reduction: int = 2,
        groups: Optional[int] = None,
        dropout_p: float = 0.1
    ):
        super().__init__()
        input_sizes = clip_model.token_c 
        self.haras = nn.ModuleList([
            hara(
                in_channels=in_ch,
                out_channels=target,
                reduction=reduction,
                groups=groups,
                dropout_p=dropout_p
            )
            for in_ch in input_sizes
        ])

    @staticmethod
    def _l2_normalize(feat: torch.Tensor, dim: int = -1, eps: float = 1e-6):
        return feat / (feat.norm(dim=dim, keepdim=True) + eps)

    def forward(self, tokens):
        vision_features = []
        for token, adapter in zip(tokens, self.haras):
            feat = adapter(token)
            feat = feat.permute(0, 2, 3, 1).contiguous()
            feat = self._l2_normalize(feat, dim=-1)
            vision_features.append(feat)
        return vision_features