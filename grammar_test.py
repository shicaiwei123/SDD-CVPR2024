import torch
import torchvision.transforms.functional as ttf
import numpy as np
import torch.nn as nn
import torchvision.transforms as tt
from PIL import Image
import PIL
import cv2
import torch.nn.functional as F
import torch.nn.functional as tnf


def std_mean_test():
    a = Image.open("/home/shicaiwei/data/ISIC/debug/ISIC_0000001.jpg").convert("RGB")

    transformer = tt.Compose([tt.Resize((224, 224))])
    a_torch = transformer(a)
    a_pil = a.resize((224, 224), resample=PIL.Image.BILINEAR)
    a_torch_numpy = np.array(a_torch)
    a_pil_numpy = np.array(a_pil)

    print(np.sum(a_torch_numpy - a_pil_numpy))

    a_cv2 = cv2.imread("/home/shicaiwei/data/ISIC/debug/ISIC_0000001.jpg", 0)
    a_cv2 = cv2.resize(a_cv2, (224, 224), interpolation=cv2.INTER_LINEAR)
    a_cv2 = cv2.cvtColor(a_cv2, cv2.COLOR_BGR2RGB)

    mean = np.array([0.5, 0.5, 0.5]) * 255
    std = np.array([0.5, 0.5, 0.5]) * 255
    mean = np.dstack((np.ones((224, 224)) * mean[0], np.ones((224, 224)) * mean[1], np.ones((224, 224)) * mean[2]))
    # mean = np.transpose(mean, (2, 0, 1))
    std = np.dstack((np.ones((224, 224)) * std[0], np.ones((224, 224)) * std[1], np.ones((224, 224)) * std[2]))
    # std = np.transpose(std, (2, 0, 1))
    # a_cv2 = np.divide(a_cv2 - mean, std)
    # a_cv2 = np.transpose(a_cv2, (2, 0, 1))

    R, G, B = cv2.split(a_cv2)
    for i in range(20):
        print(a_cv2[0, i + 200, 0])

    a = np.array(a).astype(np.float32)
    a_np = np.divide(a, 1)
    mean = np.array([0.5, 0.5, 0.5]) * 255
    std = np.array([0.5, 0.5, 0.5]) * 255
    mean = np.dstack((np.ones((224, 224)) * mean[0], np.ones((224, 224)) * mean[1], np.ones((224, 224)) * mean[2]))
    mean = np.transpose(mean, (2, 0, 1))
    std = np.dstack((np.ones((224, 224)) * std[0], np.ones((224, 224)) * std[1], np.ones((224, 224)) * std[2]))
    std = np.transpose(std, (2, 0, 1))
    a_np = np.divide(a_np - mean, std)
    a_np = np.transpose(a_np, (2, 0, 1))
    a_np = np.around(a_np, 4)
    print(1)


def transpose():
    a = np.random.random((2, 2, 3))
    b = np.transpose(a, (2, 0, 1))
    print(np.reshape(a, -1))
    print(np.reshape(b, -1))
    c = np.ones((3, 2, 2))

    for i in range(2):
        for j in range(2):
            print(a[i, j, 0])

    for i in range(2):
        for j in range(2):
            for k in range(3):
                c[k, i, j] = a[i, j, k]

    print(c)

    a_cv2 = cv2.imread("/home/shicaiwei/data/ISIC/debug/ISIC_0000001.jpg")
    a_cv2 = cv2.cvtColor(a_cv2, cv2.COLOR_BGR2RGB)
    a_cv2 = cv2.resize(a_cv2, (224, 224))
    R, G, B = cv2.split(a_cv2)


def nkd_loss_origin(logit_s, logit_t, gt_label, temp, gamma):
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

        # N*class
    N, c = logit_s.shape
    s_i = F.log_softmax(logit_s, dim=1)
    t_i = F.softmax(logit_t, dim=1)
    # N*1
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()

    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logit_s).scatter_(1, label, 1).bool()
    logit_s = logit_s[mask].reshape(N, -1)
    logit_t = logit_t[mask].reshape(N, -1)

    # N*class
    S_i = F.log_softmax(logit_s / temp, dim=1)
    T_i = F.softmax(logit_t / temp, dim=1)

    loss_non = (T_i * S_i).sum(dim=1).mean()
    loss_non = - gamma * (temp ** 2) * loss_non

    return loss_t + loss_non


def nkd_loss(logit_s, logit_t, gt_label, temp, gamma):
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

        # N*class
    N, c = logit_s.shape
    s_i = F.log_softmax(logit_s)
    t_i = F.softmax(logit_t, dim=1)
    # N*1
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()

    loss_t = - (t_t * s_t)

    mask = torch.ones_like(logit_s).scatter_(1, label, 1).bool()
    logit_s = logit_s[mask].reshape(N, -1)
    logit_t = logit_t[mask].reshape(N, -1)

    # N*class
    S_i = F.log_softmax(logit_s / temp)
    T_i = F.softmax(logit_t / temp, dim=1)

    loss_non = (T_i * S_i).sum(dim=1)
    loss_non = - gamma * (temp ** 2) * loss_non

    # print(loss_t.shape)
    # print(loss_non.shape)

    loss_t = torch.squeeze(loss_t, dim=1)
    return loss_t + loss_non





if __name__ == '__main__':
    # aa = torch.randn((8, 512))
    # bb = torch.randn((8, 512))
    # nkd_O = nkd_loss_origin(aa, bb, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).long(), 1, 1.5)
    # nkd = nkd_loss(aa, bb, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).long(), 1, 1.5)
    # print(nkd_O,nkd.mean())

    import  ast
    M="[1,2,3]"
    c=ast.literal_eval(M)
    print(1)