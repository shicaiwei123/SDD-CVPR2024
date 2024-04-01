from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152,resnet18_sdd, resnet34_sdd, resnet50_sdd, resnet101_sdd, resnet152_sdd
from .mobilenetv2 import MobileNetV2,MobileNetV2_SDD


imagenet_model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "MobileNetV2": MobileNetV2,

    "ResNet18_sdd": resnet18_sdd,
    "ResNet34_sdd": resnet34_sdd,
    "ResNet50_sdd": resnet50_sdd,
    "ResNet101_sdd": resnet101_sdd,
    "MobileNetV2_sdd": MobileNetV2_SDD,
}
