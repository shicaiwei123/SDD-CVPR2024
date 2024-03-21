import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def nkd_loss(logit_s, logit_t, gt_label, temp, gamma):
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

        # N*class
    N, c = logit_s.shape
    s_i = F.log_softmax(logit_s,dim=1)
    t_i = F.softmax(logit_t, dim=1)
    # N*1
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()

    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logit_s).scatter_(1, label, 1).bool()
    logit_s = logit_s[mask].reshape(N, -1)
    logit_t = logit_t[mask].reshape(N, -1)

    # N*class
    S_i = F.log_softmax(logit_s / temp,dim=1)
    T_i = F.softmax(logit_t / temp, dim=1)

    loss_non = (T_i * S_i).sum(dim=1).mean()
    loss_non = - gamma * (temp ** 2) * loss_non

    return loss_t + loss_non


class NKDLoss(Distiller):
    """ PyTorch version of NKD """

    def __init__(self, student, teacher, cfg):
        super(NKDLoss, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.warmup = cfg.DKD.WARMUP
        self.temp = 1.0
        self.gamma = 1.5

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_nkd = min(kwargs["epoch"] / self.warmup, 1.0) * nkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.temp,
            self.gamma,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_nkd,
        }
        return logits_student, losses_dict
