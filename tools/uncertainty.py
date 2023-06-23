import torch
import numpy as np


def activate_uce(alpha):
    return alpha / torch.sum(alpha, dim=1, keepdim=True)


def aleatoric(alpha):
    soft = activate_uce(alpha)
    max_soft, hard = soft.max(dim=1)
    return (1-max_soft[:, None, :, :])/torch.max(1-max_soft[:, None, :, :])


def dissonance(alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)

    evidence = alpha - 1
    belief = evidence / S
    dis_un = torch.zeros_like(S)
    for k in range(belief.shape[0]):
        for i in range(belief.shape[1]):
            bi = belief[k][i]
            term_Bal = 0.0
            term_bj = 0.0
            for j in range(belief.shape[1]):
                if j != i:
                    bj = belief[k][j]
                    term_Bal += bj * Bal(bi, bj)
                    term_bj += bj
            dis_ki = bi * term_Bal / (term_bj + 1e-7)
            dis_un[k] += dis_ki
    return dis_un


def Bal(b_i, b_j):
    result = 1 - torch.abs(b_i - b_j) / (b_i + b_j + 1e-7)
    return result


def softmax(x):
    if x.ndim == 4:
        return torch.softmax(x, dim=1)
    else:
        # soft = torch.softmax(x, dim=2)
        # return torch.mean(soft, dim=0)
        mean = torch.mean(x, dim=0)
        return torch.softmax(mean, dim=1)


def varep(x):
    var = torch.var(x, dim=0)

    epis = 1 - 1 / var

    return epis


def entropy(pred):
    class_num = 4
    prob = softmax(pred) + 1e-10
    e = - prob * (torch.log(prob) / np.log(class_num))
    u = torch.sum(e, dim=1, keepdim=True)

    return u


def vacuity(alpha):
    class_num = alpha.shape[1]
    S = torch.sum(alpha, dim=1, keepdim=True)
    v = class_num / torch.log(S)

    return v / torch.max(v)