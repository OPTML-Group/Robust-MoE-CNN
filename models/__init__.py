from models.resnet_cifar_ori import resnet18_cifar_ori, resnet34_cifar_ori, resnet50_cifar_ori, resnet101_cifar_ori, resnet152_cifar_ori
from models.resnet_cifar_moe import resnet18_cifar_moe, resnet34_cifar_moe, resnet50_cifar_moe, resnet101_cifar_moe, resnet152_cifar_moe
from models.wrn_cifar_ori import wrn_28_10_cifar_ori, wrn_28_4_cifar_ori, wrn_34_10_cifar_ori
from models.wrn_cifar_moe import wrn_28_4_cifar_moe, wrn_28_10_cifar_moe, wrn_34_10_cifar_moe
from models.wrn_imagenet_ori import wrn_28_10_imagenet_ori, wrn_28_4_imagenet_ori, wrn_34_10_imagenet_ori
from models.wrn_imagenet_moe import wrn_28_4_imagenet_moe, wrn_28_10_imagenet_moe, wrn_34_10_imagenet_moe
from models.resnet_imagenet_ori import resnet18_imagenet_ori, resnet34_imagenet_ori, resnet50_imagenet_ori, resnet101_imagenet_ori, resnet152_imagenet_ori
from models.resnet_imagenet_moe import resnet18_imagenet_moe, resnet34_imagenet_moe, resnet50_imagenet_moe, resnet101_imagenet_moe, resnet152_imagenet_moe
from models.vgg_ori import vgg16_ori, vgg16_bn_ori
from models.vgg_moe import vgg16_moe, vgg16_bn_moe
from models.densenet_moe import densenet_cifar_moe, densenet_imagenet_moe
from models.densenet_ori import densenet_cifar_ori, densenet_imagenet_ori
from models.layers.router import build_router
__all__ = [
    "wrn_28_10_cifar_ori",
    "wrn_28_4_cifar_ori",
    "wrn_34_10_cifar_ori",
    "resnet18_imagenet_ori",
    "resnet34_imagenet_ori",
    "resnet50_imagenet_ori",
    "resnet101_imagenet_ori",
    "resnet152_imagenet_ori",
    "resnet18_cifar_ori",
    "resnet34_cifar_ori",
    "resnet50_cifar_ori",
    "resnet101_cifar_ori",
    "resnet152_cifar_ori",
    "vgg16_ori",
    "vgg16_bn_ori",
    "resnet18_cifar_moe",
    "resnet34_cifar_moe",
    "resnet50_cifar_moe",
    "resnet101_cifar_moe",
    "resnet152_cifar_moe",
    "vgg16_moe",
    "vgg16_bn_moe",
    "resnet18_imagenet_moe",
    "resnet34_imagenet_moe",
    "resnet50_imagenet_moe",
    "resnet101_imagenet_moe",
    "resnet152_imagenet_moe",
    "wrn_28_4_imagenet_moe",
    "wrn_28_10_imagenet_moe",
    "wrn_34_10_imagenet_moe",
    "wrn_28_4_cifar_moe",
    "wrn_28_10_cifar_moe",
    "wrn_34_10_cifar_moe",
    "build_router",
]
