import math

import torch


class Block(torch.nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = torch.nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = torch.nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = torch.nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = torch.nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = torch.nn.BatchNorm2d(out_ch)
        self.bnorm2 = torch.nn.BatchNorm2d(out_ch)
        self.relu = torch.nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class TimePositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
