import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import sys

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict, cub_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
import ast
import json


def conv_bn():
    return nn.Sequential(
        nn.Conv2d(3, 16, 7, 2, 3, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2))
    )

def conv_1x1_bn(inp, oup=1280):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def modify_student_model_for_cub200(model, cfg,n_cls):
    if cfg.DISTILLER.STUDENT == 'resnet18_sdd':

        model.linear = nn.Linear(512, n_cls)

    elif cfg.DISTILLER.STUDENT == 'resnet8x4_sdd':

        model.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AvgPool2d(28)

        model.fc = nn.Linear(256, n_cls)
    elif cfg.DISTILLER.STUDENT == 'ShuffleV1_sdd':

        model.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AvgPool2d(14)

    elif cfg.DISTILLER.STUDENT == 'ShuffleV2_sdd':

        model.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AvgPool2d(14)


    elif cfg.DISTILLER.STUDENT == 'vgg8_sdd':
        model.classifier = nn.Linear(512, n_cls)

    elif cfg.DISTILLER.STUDENT == 'MobileNetV2_sdd':
        model.conv1 = conv_bn()
        model.avgpool = nn.AvgPool2d(8, ceil_mode=True)
        # print(model_s)
    else:
        raise EOFError

    return model


def modify_teacher_model_for_cub200(model,cfg,n_cls):
    if cfg.DISTILLER.TEACHER == 'resnet32x4_sdd':
        model.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AvgPool2d(28)

        model.fc = nn.Linear(256, n_cls)
    elif cfg.DISTILLER.TEACHER == 'vgg13_sdd':
        model.classifier = nn.Linear(512, n_cls)

    elif cfg.DISTILLER.TEACHER == 'ResNet50_sdd':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.linear = nn.Linear(2048, n_cls)
    else:
        raise NotImplementedError

    return model


def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)

    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)

        elif cfg.DATASET.TYPE == "cub200":

            net, pretrain_model_path = cub_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                    pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes, M=cfg.M)

            model_teacher = modify_teacher_model_for_cub200(model_teacher, cfg,n_cls=num_classes)

            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cub_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes, M=args.M
            )
            model_student = modify_student_model_for_cub200(model_student, cfg,n_cls=num_classes)

        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                    pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes, M=cfg.M)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes, M=args.M
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    distiller = nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--warmup", type=float, default=20.0)
    parser.add_argument("--M", default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.distributation = False
    cfg.warmup = args.warmup
    if args.M is not None:
        # M=ast.literal_eval(args.M)
        M = args.M
    else:
        M = args.M
    cfg.M = M
    print(type(cfg.M))
    cfg.freeze()
    main(cfg, args.resume, args.opts)
