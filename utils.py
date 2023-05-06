import torch
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import einops
import imageio
from IPython.display import clear_output

from metrics.rate_score import rate_score
from metrics.inception_score import inception_score
from metrics.fid_score import fid_score


def show_images(images, cmap='gray', title=""):
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
                plt.imshow(images[idx][0], cmap=cmap)
                idx += 1
    fig.suptitle(title, fontsize=30)

    plt.show()

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], title="Images in the first batch")
        break

def transform_data_for_show(ds_fn, train=True, store_path='../datasets'):
    transform = Compose(
        [
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2),
        ]
    )
    dataset = ds_fn(store_path, download=True, train=train, transform=transform)
    return dataset

def show_forward(ddpm, loader, device, percentiles = (0.33, 0.66, 1)):
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, title="Original images")

        for percent in percentiles:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                title=f"DDPM Noisy images {int(percent * 100)}%"
            )
        break

def training(ddpm, dataloader, n_epochs, optimizer, device, display=False, upset_epoch=10, store_path='ddpm.pt'):
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
            if x.size()[1] == 1:
                noise_est = ddpm.backward(noise, t.reshape(batch_size, -1))
            else:
                noise_est = ddpm.backward(noise, t)

            loss = loss_function(eps, noise_est)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size / len(dataloader.dataset)
        
        if display:
            show_images(generate_new_images(ddpm, device=device), title=f"Images generated at epoch {epoch + 1}")

        
        epoch_loss_history.append(epoch_loss)
        if display and epoch % upset_epoch == 0:
            clear_output(True)
            plt.figure(figsize=(16, 9))
            plt.subplot(1, 1, 1)
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

def calculate_metrics(generated_image, real_image,  device):
    inc_score, _ = inception_score(generated_image, device=device, resize=True, splits=10)
    rate_score_value = rate_score(generated_image)
    fid_score_value = fid_score(generated_image, real_image)
    return inc_score, rate_score_value.cpu().detach(), fid_score_value

def generate_new_images(ddpm, dataset=None, n_samples=16, upset=100, record_gif=True, show_metrics_pes_step=False, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=28, w=28):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []
    rate_score_history = []
    inception_score_history = []
    fid_score_history = []

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

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z
            
            
            if show_metrics_pes_step:
                real_image = next(iter(loader))[0].to(device)
                inception_score, rate_score, fid_score = calculate_metrics(x, real_image, device)
                
                rate_score_history.append(rate_score)
                inception_score_history.append(inception_score)
                fid_score_history.append(fid_score)
                
                if ddpm.n_steps - idx % upset == 0:
                    clear_output(True)
                    plt.figure(figsize=(16, 9))
                    
                    plt.subplot(1, 3, 1)
                    plt.title("Inception score")
                    plt.plot(inception_score_history)
                    plt.grid()
                    
                    plt.subplot(1, 3, 2)
                    plt.title("FID score")
                    plt.plot(fid_score_history)
                    plt.grid()
                    
                    plt.subplot(1, 3, 3)
                    plt.title("bits/dim")
                    plt.plot(rate_score_history)
                    plt.grid()

                    plt.show()
                
            if (idx in frame_idxs or t == 0) and record_gif:
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
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