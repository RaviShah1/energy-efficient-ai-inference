import torch
import torch_pruning as tp

def prune_by_importance(model, ratio=0.1):
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.GroupMagnitudeImportance(p=2) 

    ignored_layers = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if m.out_features == 100:
                ignored_layers.append(m)
            if "attention" in name:
                ignored_layers.append(m)

    pruner = tp.pruner.BasePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
        round_to=8,
    )
        
    pruner.step()