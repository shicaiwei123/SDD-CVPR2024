import os
from .resnet import (
    resnet8,
    resnet14,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
    resnet8x4,
    resnet32x4,
)

from .resnet import (
    resnet8_sdd,
    resnet14_sdd,
    resnet20_sdd,
    resnet32_sdd,
    resnet44_sdd,
    resnet56_sdd,
    resnet110_sdd,
    resnet8x4_sdd,
    resnet32x4_sdd,
)

from .resnetv2 import ResNet50, ResNet18
from .resnetv2 import ResNet50_sdd, ResNet18_sdd
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .wrn import wrn_16_1_sdd, wrn_16_2_sdd, wrn_40_1_sdd, wrn_40_2_sdd
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .vgg import vgg19_bn_sdd, vgg16_bn_sdd, vgg13_bn_sdd, vgg11_bn_sdd, vgg8_bn_sdd
from .mobilenetv2 import mobile_half, mobile_half_sdd
from .ShuffleNetv1 import ShuffleV1, ShuffleV1_sdd
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_sdd

cifar100_model_prefix = os.path.join("./save/models/"
                                     )
cifar_model_dict = {
    # teachers
    "resnet56": (
        resnet56,
        cifar100_model_prefix + "resnet56_vanilla/ckpt_epoch_240.pth",
    ),
    "resnet110": (
        resnet110,
        cifar100_model_prefix + "resnet110_vanilla/ckpt_epoch_240.pth",
    ),
    "resnet32x4": (
        resnet32x4,
        cifar100_model_prefix + "resnet32x4_vanilla/ckpt_epoch_240.pth",
    ),
    "ResNet50": (
        ResNet50,
        cifar100_model_prefix + "ResNet50_vanilla/ckpt_epoch_240.pth",
    ),
    "wrn_40_2": (
        wrn_40_2,
        cifar100_model_prefix + "wrn_40_2_vanilla/ckpt_epoch_240.pth",
    ),
    "vgg13": (vgg13_bn, cifar100_model_prefix + "vgg13_vanilla/ckpt_epoch_240.pth"),

    # sdd
    "resnet56_sdd": (
        resnet56_sdd,
        cifar100_model_prefix + "resnet56_vanilla/ckpt_epoch_240.pth",
    ),
    "resnet110_sdd": (
        resnet110_sdd,
        cifar100_model_prefix + "resnet110_vanilla/ckpt_epoch_240.pth",
    ),
    "resnet32x4_sdd": (
        resnet32x4_sdd,
        cifar100_model_prefix + "resnet32x4_vanilla/ckpt_epoch_240.pth",
    ),
    "ResNet50_sdd": (
        ResNet50_sdd,
        cifar100_model_prefix + "ResNet50_vanilla/ckpt_epoch_240.pth",
    ),
    "wrn_40_2_sdd": (
        wrn_40_2_sdd,
        cifar100_model_prefix + "wrn_40_2_vanilla/ckpt_epoch_240.pth",
    ),
    "vgg13_sdd": (vgg13_bn_sdd, cifar100_model_prefix + "vgg13_vanilla/ckpt_epoch_240.pth"),

    # students
    "resnet8": (resnet8, None),
    "resnet14": (resnet14, None),
    "resnet20": (resnet20, None),
    "resnet32": (resnet32, None),
    "resnet44": (resnet44, None),
    "resnet8x4": (resnet8x4, None),
    "ResNet18": (ResNet18, None),
    "wrn_16_1": (wrn_16_1, None),
    "wrn_16_2": (wrn_16_2, None),
    "wrn_40_1": (wrn_40_1, None),
    "wrn_16_1_sdd": (wrn_16_1_sdd, None),
    "wrn_16_2_sdd": (wrn_16_2_sdd, None),
    "wrn_40_1_sdd": (wrn_40_1_sdd, None),
    "vgg8": (vgg8_bn, None),
    "vgg11": (vgg11_bn, None),
    "vgg16": (vgg16_bn, None),
    "vgg19": (vgg19_bn, None),
    "vgg8_sdd": (vgg8_bn_sdd, None),
    "vgg11_sdd": (vgg11_bn_sdd, None),
    "vgg16_sdd": (vgg16_bn_sdd, None),
    "vgg19_sdd": (vgg19_bn_sdd, None),
    "MobileNetV2": (mobile_half, None),
    "MobileNetV2_sdd": (mobile_half_sdd, None),
    "ShuffleV1": (ShuffleV1, None),
    "ShuffleV2": (ShuffleV2, None),

    "ShuffleV1_sdd": (ShuffleV1_sdd, None),
    "ShuffleV2_sdd": (ShuffleV2_sdd, None),

    "resnet8_sdd": (resnet8_sdd, None),
    "resnet14_sdd": (resnet14_sdd, None),
    "resnet20_sdd": (resnet20_sdd, None),
    "resnet32_sdd": (resnet32_sdd, None),
    "resnet44_sdd": (resnet44_sdd, None),
    "resnet8x4_sdd": (resnet8x4_sdd, None),
    "ResNet18_sdd": (ResNet18_sdd, None),

}

cub200_model_prefix = os.path.join("./save/cub200/")

cub_model_dict = {
    "vgg13_sdd": (vgg13_bn_sdd, cub200_model_prefix + "vgg13_vanilla/vgg13_best.pth"),

    "resnet32x4_sdd": (
        resnet32x4_sdd,
        cub200_model_prefix + "resnet32x4_vanilla/resnet32x4_best.pth",
    ),
    "ResNet50_sdd": (
        ResNet50_sdd,
        cub200_model_prefix + "ResNet50_vanilla/ResNet50_best.pth",
    ),
    # students
    "resnet8": (resnet8, None),
    "resnet14": (resnet14, None),
    "resnet20": (resnet20, None),
    "resnet32": (resnet32, None),
    "resnet44": (resnet44, None),
    "resnet8x4": (resnet8x4, None),
    "ResNet18": (ResNet18, None),
    "wrn_16_1": (wrn_16_1, None),
    "wrn_16_2": (wrn_16_2, None),
    "wrn_40_1": (wrn_40_1, None),
    "wrn_16_1_sdd": (wrn_16_1_sdd, None),
    "wrn_16_2_sdd": (wrn_16_2_sdd, None),
    "wrn_40_1_sdd": (wrn_40_1_sdd, None),
    "vgg8": (vgg8_bn, None),
    "vgg11": (vgg11_bn, None),
    "vgg16": (vgg16_bn, None),
    "vgg19": (vgg19_bn, None),
    "vgg8_sdd": (vgg8_bn_sdd, None),
    "vgg11_sdd": (vgg11_bn_sdd, None),
    "vgg16_sdd": (vgg16_bn_sdd, None),
    "vgg19_sdd": (vgg19_bn_sdd, None),
    "MobileNetV2": (mobile_half, None),
    "MobileNetV2_sdd": (mobile_half_sdd, None),
    "ShuffleV1": (ShuffleV1, None),
    "ShuffleV2": (ShuffleV2, None),

    "ShuffleV1_sdd": (ShuffleV1_sdd, None),
    "ShuffleV2_sdd": (ShuffleV2_sdd, None),

    "resnet8_sdd": (resnet8_sdd, None),
    "resnet14_sdd": (resnet14_sdd, None),
    "resnet20_sdd": (resnet20_sdd, None),
    "resnet32_sdd": (resnet32_sdd, None),
    "resnet44_sdd": (resnet44_sdd, None),
    "resnet8x4_sdd": (resnet8x4_sdd, None),
    "ResNet18_sdd": (ResNet18_sdd, None),
}
