import torch
from tqdm import tqdm
from metrics import MultiClassClassificationMetrics

def evaluate_epoch(model, dataloader, device, name):
    metrics = MultiClassClassificationMetrics()
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            metrics.update(outputs.logits, labels)
    
    print(f"== {name} Metrics ==")
    metrics.report()
