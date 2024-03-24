import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .DKD import dkd_loss as dkd_loss_origin


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='none')
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none')
            * (temperature ** 2)
            / target.shape[0]
    )

    tckd_loss = torch.sum(tckd_loss, dim=1)
    nckd_loss = torch.sum(nckd_loss, dim=1)
    return alpha * tckd_loss + beta * nckd_loss


def multi_dkd(out_s_multi, out_t_multi, target, alpha, beta, temperature):
    ###############################shape convert######################
    #  from B X C X N to N*B X C. Here N is the number of decoupled region
    #####################


    out_s_multi_t = out_s_multi.permute(2, 0, 1)
    out_t_multi_t = out_t_multi.permute(2, 0, 1)

    out_t = torch.reshape(out_t_multi_t, (out_t_multi_t.shape[0] * out_t_multi_t.shape[1], out_t_multi_t.shape[2]))
    out_s = torch.reshape(out_s_multi_t, (out_s_multi_t.shape[0] * out_s_multi_t.shape[1], out_s_multi_t.shape[2]))
    # print(out_s.shape)
    target_r = target.repeat(out_t_multi.shape[2])


    ####################### calculat distillation loss##########################
    # only conduct average or sum in the dim of calss and skip the dim of batch

    loss = dkd_loss(out_s, out_t, target_r, alpha, beta, temperature)




    ######################find the complementary and consistent local distillation loss#############################


    out_t_predict = torch.argmax(out_t, dim=1)

    mask_true = out_t_predict == target_r
    mask_false = out_t_predict != target_r



    # global_prediction = out_t_predict[len(target_r) - len(target):len(target_r)]
    global_prediction = out_t_predict[0:len(target)]
    global_prediction_true_mask = global_prediction == target
    global_prediction_false_mask = global_prediction != target

    global_prediction_true_mask_repeat = torch.tensor(global_prediction_true_mask).repeat(out_t_multi.shape[2])
    global_prediction_false_mask_repeat = torch.tensor(global_prediction_false_mask).repeat(out_t_multi.shape[2])

    # global true local worng
    mask_false[global_prediction_false_mask_repeat] = False
    # mask_false[len(target_r) - len(target):len(target_r)] = False
    mask_false[0:len(target)] = False

    gt_lw = mask_false

    # global wrong local true

    mask_true[global_prediction_true_mask_repeat] = False
    # mask_true[len(target_r) - len(target):len(target_r)] = False
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

    assert torch.sum(gt_lt) + torch.sum(gw_lw) + torch.sum(gt_lw) + torch.sum(gw_lt)==target_r.shape[0]

    ########################################Modify the weight of complementary terms#######################

    index[gw_lw] = 1.0
    index[gt_lt] = 1.0
    index[gw_lt] = 2.0
    index[gt_lw] = 2.0


    loss = torch.sum(loss * index)

    if torch.isnan(loss) or torch.isinf(loss):
        print("inf")
        loss = torch.zeros(1).float().cuda()

    return loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class SDD_DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(SDD_DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.warmup
        self.M=cfg.M

    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t = self.teacher(image)

        # losses
        # print(self.warmup)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        if self.M=='[1]':
            loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss_origin(
                logits_student,
                logits_teacher,
                target,
                self.alpha,
                self.beta,
                self.temperature,
            )
        else:
            loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * multi_dkd(
                patch_s,
                patch_t,
                target,
                self.alpha,
                self.beta,
                self.temperature,
            )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
