import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from IPython.display import clear_output
import numpy as np

from metrics.fid_score import fid_score
from metrics.inception_score import inception_score
from metrics.rate_score import rate_score


def show_images(images, cmap="gray", title=""):
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

def show_tensor_images(images, idx=0):
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    if len(images.shape) == 4:
        image = images[idx, :, :, :].detach().cpu()
    
    plt.imshow(reverse_transform(image))


def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], title="Images in the first batch")
        break


def transform_data(ds_fn, image_size=(32, 32), train=True, store_path="../datasets"):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    dataset = ds_fn(store_path, download=True, train=train, transform=transform)
    return dataset


def show_forward(ddpm, loader, device, percentiles=(0.33, 0.66, 1)):
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, title="Original images")

        for percent in percentiles:
            show_images(
                ddpm(
                    imgs.to(device),
                    [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))],
                ),
                title=f"DDPM Noisy images {int(percent * 100)}%",
            )
        break


def calculate_metrics(generated_image, real_image, device):
    inc_score, _ = inception_score(
        generated_image, device=device, resize=True, splits=10
    )
    fid_score_value = fid_score(generated_image, real_image, device)
    return inc_score, fid_score_value

def plot_metrics_iddpm(generated_images, real_images, device, n_timestamps=250, batch_size=100):
    inception_score_history = []
    fid_score_history = []
    real_dataloader = DataLoader(real_images, batch_size=batch_size, num_workers=0, shuffle=True)
    
    for timestamp in tqdm(range(n_timestamps)):
        gen_batch_t = generated_images[:, timestamp, :, :].to(device)
        real_batch_t = next(iter(real_dataloader))[0].to(device)
        
        fid_score_value = fid_score(gen_batch_t, real_batch_t, batch_size=batch_size, device=device)
        inception_score_value, _ = inception_score(gen_batch_t, device=device, batch_size=batch_size, resize=True)
        
        fid_score_history.append(fid_score_value)
        inception_score_history.append(inception_score_value)
    
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
