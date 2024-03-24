from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample, get_imagenet_dataloaders_distribution
from .cub200 import get_cub200_dataloaders


def get_dataset(cfg):
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 100

    elif cfg.DATASET.TYPE == "cub200":
        train_loader, val_loader, num_data = get_cub200_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            is_instance=True
        )

        num_classes = 200


    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            if cfg.distributation:
                train_loader, val_loader, num_data = get_imagenet_dataloaders_distribution(
                    batch_size=cfg.SOLVER.BATCH_SIZE,
                    val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                    num_workers=cfg.DATASET.NUM_WORKERS,
                )
            else:
                train_loader, val_loader, num_data = get_imagenet_dataloaders(
                    batch_size=cfg.SOLVER.BATCH_SIZE,
                    val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                    num_workers=cfg.DATASET.NUM_WORKERS,
                )
        num_classes = 1000
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes
