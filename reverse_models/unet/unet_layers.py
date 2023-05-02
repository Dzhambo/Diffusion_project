import torch
import math


class TimeEmbedding(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = torch.nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = torch.nn.SiLU()
        self.lin2 = torch.nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_channels,
                 n_groups = 32, dropout = 0.1):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(n_groups, in_channels)
        self.act1 = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.norm2 = torch.nn.GroupNorm(n_groups, out_channels)
        self.act2 = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = torch.nn.Identity()

        self.time_emb = torch.nn.Linear(time_channels, out_channels)
        self.time_act = torch.nn.SiLU()

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(torch.nn.Module):
    def __init__(self, n_channels, n_heads = 1, d_k = None, n_groups = 32):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = torch.nn.GroupNorm(n_groups, n_channels)
        self.projection = torch.nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = torch.nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x, t = None):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = torch.nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = torch.nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(torch.nn.Module):
    def __init__(self, n_channels, time_channels):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x, t):
        return self.conv(x)


class Downsample(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x, t):
        return self.conv(x)
