import torch


class CustomDiffusionModel(torch.nn.Module):
    def __init__(
        self,
        network,
        n_steps=200,
        min_beta=10**-4,
        max_beta=0.02,
        device=None,
        beta_schedule="linear",
    ):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.betas = self.get_beta_schedule(beta_schedule, min_beta, max_beta)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]
        ).to(device)

    def _warmup_beta(self, beta_start, beta_end, timesteps, warmup_frac, dtype):
        betas = beta_end * torch.ones(timesteps, dtype=dtype)
        warmup_time = int(timesteps * warmup_frac)
        betas[:warmup_time] = torch.linspace(
            beta_start, beta_end, warmup_time, dtype=dtype
        )
        return betas

    def get_beta_schedule(
        self, beta_schedule, beta_start, beta_end, timesteps, dtype=torch.float64
    ):
        if beta_schedule == "quad":
            betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, timesteps, dtype=dtype
                )
                ** 2
            )
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)
        elif beta_schedule == "warmup10":
            betas = self._warmup_beta(beta_start, beta_end, timesteps, 0.1, dtype=dtype)
        elif beta_schedule == "warmup50":
            betas = self._warmup_beta(beta_start, beta_end, timesteps, 0.5, dtype=dtype)
        elif beta_schedule == "const":
            betas = beta_end * torch.ones(timesteps, dtype=dtype)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps, dtype=dtype)
        else:
            # Дописать cosine scheduler
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (timesteps,)
        return betas

    def forward(self, x, t, eps=None):
        b, c, h, w = x.shape
        a_bar = self.alpha_bars[t]

        if eps is None:
            eps = torch.randn(b, c, h, w).to(self.device)

        noise = (
            a_bar.sqrt().reshape(b, 1, 1, 1) * x
            + (1 - a_bar).sqrt().reshape(b, 1, 1, 1) * eps
        )
        return noise

    def backward(self, x, t):
        return self.network(x, t)
