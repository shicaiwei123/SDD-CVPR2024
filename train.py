

import sys

sys.path.append('..')

# os.environ["OMP_NUM_THREADS"] = str(1)


import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# cudnn.benchmark = True
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as tm


def init_distributed_mode(args):
    '''initilize DDP
    '''
    print("innnnnnnnnnnnnnnnn")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print(111111111111)
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        print(2222222222222)
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        print(3333333333333333)
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}, local rank:{args.gpu}, world size:{args.world_size}",
        flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )


def main(cfg, resume, opts, distribution_arsg):
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

    # init distibuation

    # print(type(cfg.local_rank))
    # local_rank = cfg.local_rank
    # device = torch.device("cuda", int(local_rank))
    #
    # # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    # # torch.cuda.set_device(local_rank)
    # #   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
    # dist.init_process_group(backend='nccl')
    # print('world_size', torch.distributed.get_world_size())

    init_distributed_mode(distribution_arsg)
    use_cuda = torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

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
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True,M=cfg.M)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False,M=cfg.M)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                    pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes,M=cfg.M)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes,M=cfg.M
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    distiller = torch.nn.SyncBatchNorm.convert_sync_batchnorm(distiller)
    distiller = distiller.to(device)
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    # print(device, local_rank, torch.cuda.is_available())

    # model= tm.resnet18(pretrained=False)
    # model=model.to(device)

    distiller = DDP(distiller, device_ids=[distribution_arsg.gpu], find_unused_parameters=True)

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

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="configs/imagenet/r50_mv2/sdd_dkd.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    # parser.add_argument("--gpu", default=0)
    parser.add_argument('--local_rank', type=int, help='local rank, will passed by ddp')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--M", default='[1,2,4]')


    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.local_rank = args.local_rank
    cfg.distributation = True
    cfg.M = args.M
    cfg.warmup=1.0

    cfg.freeze()
    main(cfg, args.resume, args.opts, args)
