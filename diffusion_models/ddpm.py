import torch


class CustomDiffusionModel(torch.nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device) 
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x, t, eps=None):
        b, c, h, w = x.shape
        a_bar = self.alpha_bars[t]

        if eps is None:
            eps = torch.randn(b, c, h, w).to(self.device)

        noise = a_bar.sqrt().reshape(b, 1, 1, 1) * x + (1 - a_bar).sqrt().reshape(b, 1, 1, 1) * eps
        return noise

    def reverse(self, x, t):
        return self.network(x, t)
