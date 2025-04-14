import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch_pruning as tp


def model_analysis(model, shape=(1, 3, 224, 224)):
    dummy_input = torch.randn(shape)
    flops, nparams = tp.utils.count_ops_and_params(model, dummy_input)

    print(f"Total FLOPs  : {flops / 1e9:.2f} G")
    print(f"Total Params : {nparams / 1e6:.2f} M")


def flop_analysis(model, shape=(1, 3, 224, 224)):
    dummy_input = torch.randn(shape)
    flops = FlopCountAnalysis(model, dummy_input)
    flop_table = flop_count_table(flops, max_depth=4)
    print(flop_table)