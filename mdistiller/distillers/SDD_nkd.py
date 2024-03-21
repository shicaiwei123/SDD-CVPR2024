import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


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


def multi_nkd_loss(logit_s, logit_t, gt_label, temp, gamma):

    ###############################shape convert######################
    #  from B X C X N to N*B X C. Here N is the number of decoupled region
    #####################

    out_s_multi_t = logit_s.permute(2, 0, 1)
    out_t_multi_t = logit_t.permute(2, 0, 1)

    out_t = torch.reshape(out_t_multi_t, (out_t_multi_t.shape[0] * out_t_multi_t.shape[1], out_t_multi_t.shape[2]))
    out_s = torch.reshape(out_s_multi_t, (out_s_multi_t.shape[0] * out_s_multi_t.shape[1], out_s_multi_t.shape[2]))
    # print(out_s.shape)
    target_r = gt_label.repeat(logit_t.shape[2])

    ####################### calculat distillation loss##########################
    # only conduct average or sum in the dim of calss and skip the dim of batch

    loss = nkd_loss(out_s, out_t, target_r, temp, gamma)


    ######################find the complementary and consistent local distillation loss#############################


    out_t_predict = torch.argmax(out_t, dim=1)

    mask_true = out_t_predict == target_r
    mask_false = out_t_predict != target_r


    target = gt_label

    global_prediction = out_t_predict[0:len(target)]
    global_prediction_true_mask = global_prediction == target
    global_prediction_false_mask = global_prediction != target

    global_prediction_true_mask_repeat = torch.tensor(global_prediction_true_mask).repeat(logit_t.shape[2])
    global_prediction_false_mask_repeat = torch.tensor(global_prediction_false_mask).repeat(logit_t.shape[2])

    # global true local worng
    mask_false[global_prediction_false_mask_repeat] = False
    mask_false[0:len(target)] = False

    gt_lw = mask_false

    # global wrong local true

    mask_true[global_prediction_true_mask_repeat] = False
    mask_true[0:len(target)] = False

    gw_lt = mask_true

    mask_false = out_t_predict != target_r
    mask_true = out_t_predict == target_r

    index = torch.zeros_like(loss).float()



    # global wrong local wrong
    mask_false[global_prediction_true_mask_repeat] = False
    gw_lw = mask_false

    mask_true[global_prediction_false_mask_repeat] = False
    gt_lt = mask_true

    # print(torch.sum(gt_lt) + torch.sum(gw_lw) + torch.sum(gt_lw) + torch.sum(gw_lt))
    ########################################Modify the weight of complementary terms#######################

    index[gw_lw] = 1.0
    index[gt_lt] = 1.0
    index[gw_lt] = 2
    index[gt_lw] = 2

    index = torch.tensor(index)

    loss = torch.sum(loss * index) / target_r.shape[0]


    # print(loss)
    if torch.isnan(loss) or torch.isinf(loss):
        print("inf")
        loss = torch.zeros(1).float().cuda()

    return loss


class SDD_NKDLoss(Distiller):
    """ PyTorch version of NKD """

    def __init__(self, student, teacher, cfg):
        super(SDD_NKDLoss, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.warmup = cfg.DKD.WARMUP
        self.temp = cfg.DKD.T
        self.gamma = cfg.DKD.BETA
        self.M=cfg.M

    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * 1.0 * F.cross_entropy(logits_student, target)
        if self.M=='[1]':
            # print("M1111111111")
            loss_nkd=min(kwargs["epoch"] / self.warmup, 1.0) * nkd_loss_origin(
                logits_student,
                logits_teacher,
                target,
                self.temp,
                self.gamma,
            )
        else:
            loss_nkd = min(kwargs["epoch"] / self.warmup, 1.0) * multi_nkd_loss(
                patch_s,
                patch_t,
                target,
                self.temp,
                self.gamma,
            )

        # loss_nkd = min(kwargs["epoch"] / self.warmup, 1.0) * nkd_loss_origin(
        #     logits_student,
        #     logits_teacher,
        #     target,
        #     self.temp,
        #     self.gamma,
        # )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_nkd,
        }

        # print(11111)
        return logits_student, losses_dict
