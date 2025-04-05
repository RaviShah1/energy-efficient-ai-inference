import torch
import torch.nn.utils.prune as prune

def prune_by_masking(model, ratio=0.1):
    parameters_to_prune = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.out_features == 100:
                continue
            if "attention" in name:
                continue
            parameters_to_prune.append((module, 'weight'))

    # Apply global unstructured L1 pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=ratio
    )

    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')