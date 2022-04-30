import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
import numpy as np


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        split_num = 4

        num_patches = split_num ** 3
        patch_dim = int(8000 / num_patches) * 5
        self.pool = 'mean'

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
        )
        self.m = nn.AdaptiveAvgPool1d(1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.m(x.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x).reshape(b, 50, 50)


class Custom_LossFunction(nn.Module):
    def __init__(self):
        super(Custom_LossFunction, self).__init__()

    def forward(self, Est, Truth):
        device = Est.device
        b = Est.shape[0]
        Est = Est.flatten(1)
        Truth = Truth.flatten(1)
        # Step1 & 2: Calculate residual between Truth and Est and convert them to [0,2*pi]
        # TwoPi = np.ones((BATCH_SIZE, 2500)) * 2 * np.pi
        # TwoPi = torch.from_numpy(TwoPi).to(device)
        Residual_abs = torch.abs(Truth - Est)
        Residual = 2 * np.pi * Residual_abs   # [0, 2pi]

        # print('minimum Residual between Truth and Est: {:4f}'.format(Residual.min()))
        # print('maximum Residual between Truth and Est: {:4f}'.format(Residual.max()))

        # Step 3: Calculate the cosin value of the residual
        Residual_cos = torch.cos(Residual)  # [-1, 1]
        # print('minimum Cosin Residual between Truth and Est: {:4f}'.format(Residual_cos.min()))
        # print('maximum Cosin Residual between Truth and Est: {:4f}'.format(Residual_cos.max()))

        # Step 4: Calculate I - Residual_cos
        Residual_cos_size = list(Residual_cos.size())
        I = np.ones((Residual_cos_size[0], 2500))
        I = torch.from_numpy(I).to(device)
        Distance = I - Residual_cos  # [0, 2]
        Distance_min = Distance.min()
        Distance_max = Distance.max()
        # print('minimum Distance between Truth and Est: {:4f}'.format(Distance_min))
        # print('maximum Distance between Truth and Est: {:4f}'.format(Distance_max))

        # Step 5: Calculate loss value
        Loss = torch.mean(Distance)

        return Loss
