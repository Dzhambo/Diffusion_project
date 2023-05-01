import torch
from reverse_models.unet_layers import Block, TimePositionEmbeddings


class CustomUnet(torch.nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.image_channels = image_channels
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = self.image_channels
        time_emb_dim = 32

        self.time_mlp = torch.nn.Sequential(
                TimePositionEmbeddings(time_emb_dim),
                torch.nn.Linear(time_emb_dim, time_emb_dim),
                torch.nn.ReLU()
            )
        
        self.conv0 = torch.nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = torch.nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        
        self.ups = torch.nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = torch.nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)
