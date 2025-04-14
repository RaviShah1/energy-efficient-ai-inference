import torch
import torch_pruning as tp
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTSelfOutput
 

def prune_by_importance(model, ratio=0.05, include_attn=False):
    if include_attn:
        example_inputs = torch.randn(1, 3, 224, 224)
        imp = tp.importance.MagnitudeImportance(p=1)

        num_heads = {}
        ignored_layers = [model.classifier]
        for m in model.modules():
            if isinstance(m, ViTSelfAttention):
                num_heads[m.query] = m.num_attention_heads
                num_heads[m.key] = m.num_attention_heads
                num_heads[m.value] = m.num_attention_heads

        pruner = tp.pruner.BasePruner(
                    model, 
                    example_inputs, 
                    global_pruning=False,
                    importance=imp,
                    pruning_ratio=ratio,
                    ignored_layers=ignored_layers,
                    output_transform=lambda out: out.logits.sum(),
                    num_heads=num_heads,
                    prune_head_dims=True,
                    prune_num_heads=False,
                    round_to=4,
        )

        for g in pruner.step(interactive=True):
            g.prune()

        for m in model.modules():
            if isinstance(m, ViTSelfAttention):
                m.num_attention_heads = pruner.num_heads[m.query]
                m.attention_head_size = m.query.out_features // m.num_attention_heads
                m.all_head_size = m.query.out_features
    else:
        example_inputs = torch.randn(1, 3, 224, 224)
        imp = tp.importance.GroupMagnitudeImportance(p=1) 

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
            round_to=4,
        )
        
        pruner.step()