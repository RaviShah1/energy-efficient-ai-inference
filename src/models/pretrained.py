import torch
from transformers import ViTForImageClassification, ViTConfig
from quanto import quantize, freeze, qint8, qint4, qfloat8


def get_pretrained_vit(quantization: bool=True):
    config = ViTConfig.from_pretrained('edumunozsala/vit_base-224-in21k-ft-cifar100')
    model = ViTForImageClassification.from_pretrained('edumunozsala/vit_base-224-in21k-ft-cifar100',
                                                      config=config)
    if quantization:
        quantize(model, weights=qint8, activations=None)
        freeze(model)
    return model, config