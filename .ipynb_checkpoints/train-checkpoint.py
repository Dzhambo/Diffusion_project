import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from IPython.display import clear_output
from torch.optim import Adam
from tqdm import tqdm

from diffusion_models.ddpm_classifier_free_guidance import DDPM, ddpm_schedules, ContextUnet


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
        
    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...').cpu().numpy()

        mu = np.mean(features, axis = 0)
        sigma = np.cov(features, rowvar = False)
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):

        if self.channels == 1:
            real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (real_samples, fake_samples))

        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def train(self):
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
    
    def train_guidance_free(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                
                data, labels = next(iter(self.dl))[0].to(self.device), next(iter(self.dl))[1].to(self.device)

                loss = self.model(data, classes = labels)
                total_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f"loss: {total_loss:.4f}")

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                pbar.update(1)
                

                
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

        
def train_mnist(
    n_epoch=20,
    batch_size = 256,
    n_T = 400,
    n_classes=10,
    n_feat = 128,
    lrate = 1e-4,
    save_model = False,
    save_dir = '../../pictures/',
    device = "cuda:0",
    ws_test = (0.0, 0.5, 2.0)
    
):

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MNIST("../../datasets/", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%5==0 or ep == int(n_epoch-1):
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")
