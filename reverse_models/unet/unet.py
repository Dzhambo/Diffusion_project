import torch
from reverse_models.unet.unet_layers import TimeEmbedding, UpBlock, Upsample, DownBlock, Downsample, MiddleBlock


class Unet(torch.nn.Module):
    def __init__(self,
                image_channels = 3,
                ch_mults = (1, 2, 2, 4),
                n_channels = 64,
                n_blocks = 2,
                is_attn = (False, False, True, True),
                ):
        super().__init__()
        n_resolutions = len(ch_mults)
        self.image_proj = torch.nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        self.time_emb = TimeEmbedding(n_channels * 4)

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = torch.nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = torch.nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = torch.nn.GroupNorm(8, n_channels)
        self.act = torch.nn.SiLU()
        self.final = torch.nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x, t):
        t = self.time_emb(t)
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                print(x.size(), s.size())
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        return self.final(self.act(self.norm(x)))