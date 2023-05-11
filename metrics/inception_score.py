import numpy as np
import torch
from scipy.stats import entropy
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3


def get_pred(x, resize, up, inception_model):
    if resize:
        x = up(x)
    x = inception_model(x)
    return F.softmax(x).data.cpu().numpy()


def inception_score(imgs, device, batch_size=32, resize=False, splits=10):
    N = len(imgs)

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode="bilinear").to(device)

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.to(device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(
            batchv, resize, up, inception_model
        )

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
