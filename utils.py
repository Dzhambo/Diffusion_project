import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import einops
import imageio
import cv2
from IPython.display import clear_output


def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    idx = 0
    for _ in range(rows):
        for _ in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0])
                idx += 1
    fig.suptitle(title, fontsize=30)

    plt.show()

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break

def transform_data_for_show(ds_fn, store_path='../datasets'):
    transform = Compose(
        [
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2),
        ]
    )
    dataset = ds_fn("./datasets", download=True, train=True, transform=transform)

    return dataset

def show_forward(ddpm, loader, device):
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break

def training(ddpm, dataloader, n_epochs, optimizer, device, display=False, upset_epoch=100, store_path='ddpm.pt'):
    loss_function = torch.nn.MSELoss()
    best_loss = float('inf')
    epoch_loss_history = []
     
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500"):
            x = batch[0].to(device)

            batch_size = len(x)

            t = torch.randint(0, ddpm.n_steps, (batch_size,)).to(device)
            eps = torch.randn_like(x).to(device)

            noise = ddpm(x, t, eps)
            noise_est = ddpm.reverse(noise, t)

            loss = loss_function(noise, noise_est)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size / len(dataloader.dataset)
        
        epoch_loss_history.append(epoch_loss)
        if display and epoch % upset_epoch == 0:
            clear_output(True)
            plt.figure(figsize=(16, 9))
            plt.subplot(2, 1, 1)
            plt.title("MSE LOSS")
            plt.plot(epoch_loss_history)
            plt.grid()

            plt.show()
        
        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.5f}"
        

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)


def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=3, h=32, w=32):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.reverse(x, time_tensor.squeeze(1))

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

            if idx in frame_idxs or t == 0:
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                frames.append(frame)

    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])
    return x