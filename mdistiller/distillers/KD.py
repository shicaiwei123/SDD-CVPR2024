import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


def sdd_kd_loss(out_s_multi, out_t_multi, T, target):

     ###############################shape convert######################
        #  from B X C X N to N*B X C. Here N is the number of decoupled region
     #####################

    out_s_multi = out_s_multi.permute(2, 0, 1)
    out_t_multi = out_t_multi.permute(2, 0, 1)

    out_t = torch.reshape(out_t_multi, (out_t_multi.shape[0] * out_t_multi.shape[1], out_t_multi.shape[2]))
    out_s = torch.reshape(out_s_multi, (out_s_multi.shape[0] * out_s_multi.shape[1], out_s_multi.shape[2]))

    target_r = target.repeat(out_t_multi.shape[0])


    ####################### calculat distillation loss##########################

    p_s = F.log_softmax(out_s / T, dim=1)
    p_t = F.softmax(out_t / T, dim=1)
    loss_kd = F.kl_div(p_s, p_t, reduction='none') * (T ** 2)
    nan_index = torch.isnan(loss_kd)
    loss_kd[nan_index] = torch.tensor(0.0).cuda()


     # only conduct average or sum in the dim of calss and skip the dim of batch
    loss_kd = torch.sum(loss_kd, dim=1)


    ######################find the complementary and consistent local distillation loss#############################

    out_t_predict = torch.argmax(out_t, dim=1)

    mask_true = out_t_predict == target_r
    mask_false = out_t_predict != target_r

    # global_prediction = out_t_predict[len(target_r) - len(target):len(target_r)]
    global_prediction = out_t_predict[0:len(target)]
    global_prediction_true_mask = global_prediction == target
    global_prediction_false_mask = global_prediction != target

    global_prediction_true_mask_repeat = torch.tensor(global_prediction_true_mask).repeat(out_t_multi.shape[0])
    global_prediction_false_mask_repeat = torch.tensor(global_prediction_false_mask).repeat(out_t_multi.shape[0])

    # global true local worng
    mask_false[global_prediction_false_mask_repeat] = False
    mask_false[0:len(target)] = False

    gt_lw = mask_false

    # global wrong local true

    mask_true[global_prediction_true_mask_repeat] = False
    # mask_true[len(target_r) - len(target):len(target_r)] = False
    mask_true[0:len(target)] = False

    gw_lt = mask_true

    mask_false = out_t_predict != target_r
    mask_true = out_t_predict == target_r

    index = torch.zeros_like(loss_kd).float()

    # regurilize for similar


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

    loss = torch.sum(loss_kd * index) / target_r.shape[0]

    if torch.isnan(loss) or torch.isinf(loss):
        print("inf")
        loss = torch.zeros(1).float().cuda()

    return loss


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict


class SDD_KD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(SDD_KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.warmup = cfg.warmup


    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t = self.teacher(image)

        # losses
        # *min(kwargs["epoch"] / self.warmup, 1.0)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight *sdd_kd_loss(
            patch_s, patch_t, self.temperature, target
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
