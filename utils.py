import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import einops
import imageio


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

def transform_data_for_show(ds_fn, batch_size):
    transform = Compose(
        [
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2),
        ]
    )
    dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    return loader

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
