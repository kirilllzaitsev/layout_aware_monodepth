import torch
import torch.nn as nn
from einops import einsum, rearrange


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads, head_channel, dropout=0.0):
        super().__init__()
        inner_dim = heads * head_channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = head_channel**-0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, head_channel), nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, c, h, w = x.shape
        # q =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        # k = self.to_k(z).view(b, self.heads, m, c)
        # v = self.to_v(z).view(b, self.heads, m, c)
        # dots = q @ k.transpose(2, 3) * self.scale
        z = rearrange(z, "b c h w -> b (h w) c")
        q = rearrange(x, "b c h w -> b (h w) c").unsqueeze(1)
        k = rearrange(self.to_k(z), "b m (h c) -> b h m c", h=self.heads)
        v = rearrange(self.to_v(z), "b m (h c) -> b h m c", h=self.heads)
        dots = einsum(q, k, "b h l c, b h m c -> b h l m") * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, "b h l c -> b l (h c)")
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads, head_channel, dropout=0.0):
        super().__init__()
        inner_dim = heads * head_channel
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.scale = head_channel**-0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, head_channel), nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        q, k, v = torch.chunk(self.to_qkv(x), 3, dim=-1)
        dots = einsum(q, k, "b h l c, b h m c -> b h l m") * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, "b h l c -> b l (h c)")
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out
