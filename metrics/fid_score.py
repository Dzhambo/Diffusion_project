import torch
import numpy as np
from torchvision.models.inception import inception_v3


def fid_score(images1, images2, device):

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    embs1 = inception_model.predict(images1)
    embs2 = inception_model.predict(images2)

    mu1, sigma1 = embs1.mean(dim=0), torch.cov(embs1)
    mu2, sigma2 = embs2.mean(dim=0), torch.cov(embs2)

    fid_score = torch.nn.MSELoss()(mu1, mu2) + torch.trace(sigma1 + sigma2 - 2 * torch.sqrt(sigma1 * sigma2))

    return fid_score
