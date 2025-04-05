import torch

def warm_up_processors(model, input_shape, device, num_batches=5, batch_size=8):
    model.eval()

    with torch.no_grad():
        dummy_input = torch.randn(batch_size, *input_shape).to(device)

        for _ in range(num_batches):
            _ = model(dummy_input)