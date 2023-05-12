import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from IPython.display import clear_output
from torch.optim import Adam
from tqdm import tqdm


def exists(x):
    return x is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataloader,
        device,
        display=True,
        train_lr=1e-4,
        train_num_steps=100000,
        adam_betas=(0.9, 0.99),
        upset_step=1000,
    ):
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.device = device

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.dl = dataloader
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        self.step = 0
        
        self.display = display
        self.upset_step = upset_step

    def train(self):
        loss_history = []
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                
                data = next(iter(self.dl))[0].to(self.device)

                loss = self.model(data)
                total_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f"loss: {total_loss:.4f}")

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                pbar.update(1)
                
                loss_history.append(loss)
                if self.display and self.step % self.upset_step == 0:
                    clear_output(True)
                    plt.figure(figsize=(16, 9))
                    plt.subplot(1, 1, 1)
                    plt.title("LOSS")
                    plt.plot(epoch_loss_history)
                    plt.grid()

                    plt.show()
                
    print("training complete")


def training_for_ddpm(
    ddpm,
    dataloader,
    n_epochs,
    optimizer,
    device,
    display=False,
    upset_epoch=10,
    store_path="ddpm.pt",
):
    loss_function = torch.nn.MSELoss()
    best_loss = float("inf")
    epoch_loss_history = []

    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0.0
        for batch in tqdm(
            dataloader,
            leave=False,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            colour="#005500",
        ):
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
