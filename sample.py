import einops
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from torch.utils.data import DataLoader

from utils import calculate_metrics


def generate_new_images(
    ddpm,
    dataset=None,
    n_samples=16,
    upset=100,
    record_gif=True,
    show_metrics_pes_step=False,
    device=None,
    frames_per_gif=100,
    gif_name="sampling.gif",
    c=1,
    h=28,
    w=28,
):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []
    inception_score_history = []
    fid_score_history = []

    if dataset is not None:
        loader = DataLoader(dataset, batch_size=n_samples, num_workers=0, shuffle=True)

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            if c == 1:
                eta_theta = ddpm.backward(x, time_tensor)
            else:
                eta_theta = ddpm.backward(x, time_tensor.squeeze(1))

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

            if show_metrics_pes_step:
                real_image = next(iter(loader))[0].to(device)
                inception_score, fid_score = calculate_metrics(x, real_image, device)

                inception_score_history.append(inception_score)
                fid_score_history.append(fid_score)

                if ddpm.n_steps - idx % upset == 0:
                    clear_output(True)
                    plt.figure(figsize=(16, 9))

                    plt.subplot(1, 2, 1)
                    plt.title("Inception score")
                    plt.plot(inception_score_history)
                    plt.grid()

                    plt.subplot(1, 2, 2)
                    plt.title("FID score")
                    plt.plot(fid_score_history)
                    plt.grid()

                    plt.show()

            if (idx in frame_idxs or t == 0) and record_gif:
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                frame = einops.rearrange(
                    normalized,
                    "(b1 b2) c h w -> (b1 h) (b2 w) c",
                    b1=int(n_samples**0.5),
                )
                frame = frame.cpu().numpy().astype(np.uint8)

                frames.append(frame)
    if c == 3 and record_gif:
        with imageio.get_writer(gif_name, mode="I") as writer:
            for idx, frame in enumerate(frames):
                writer.append_data(frame)
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(frames[-1])

    return x
