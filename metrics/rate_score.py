import numpy as np
import torch


def rate_score(x):
    b, c, h, w = x.size()
    log_softmax = torch.nn.LogSoftmax(dim=0)
    return -torch.sum(log_softmax(x)) / (b * c * h * w * np.log(2))
