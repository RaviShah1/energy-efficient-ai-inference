import torch
from transformers import ViTForImageClassification, ViTConfig


def get_pretrained_vit():
    config = ViTConfig.from_pretrained('edumunozsala/vit_base-224-in21k-ft-cifar100')
    model = ViTForImageClassification.from_pretrained('edumunozsala/vit_base-224-in21k-ft-cifar100', config=config)
    return model, config