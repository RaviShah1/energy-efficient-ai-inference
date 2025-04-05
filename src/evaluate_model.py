import torch
import time
from tqdm import tqdm
import argparse

from data.dataset import get_test_loader
from energy.energy_tracker import EnergyTracker
from models.pretrained import get_pretrained_vit
from models.pruning.structural_pruning import prune_by_importance
from models.pruning.unstructural_pruning import prune_by_masking
from metrics import MultiClassClassificationMetrics
from utils import warm_up_processors

# export PYTHONPATH=$(pwd)/src


device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print(f"Running on device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pruning_type",
    type=str,
    default=None,
    help="Type of pruning to apply (structural, unstructural). Default is None (no pruning)."
)
parser.add_argument(
    "--pruning_ratio",
    type=float,
    default=0.1,
    help="Pruning ratio (0.0–1.0)"
)

args = parser.parse_args()


# Evaluation function with energy tracking
def evaluate(model, dataloader, device, metrics):
    model.eval()
    total = 0
    start_time = time.time()

    energy = EnergyTracker()
    energy.start()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            inputs = images.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            metrics.update(outputs.logits, labels)
            total += labels.size(0)

    usage = energy.end()
    energy.shutdown()

    total_time = time.time() - start_time
    avg_time_per_sample = total_time / total

    metrics.report()
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Average Inference Time per Sample: {avg_time_per_sample:.5f} seconds")
    print(f"Total GPU Energy Consumption: {usage['gpu_energy_joules']:.2f} Joules")
    print(f"Average GPU Energy per Sample: {usage['gpu_energy_joules'] / total:.5f} J/sample")
    print(f"Total CPU Energy Consumption: {usage['cpu_energy_joules']:.2f} Joules")
    print(f"Average CPU Energy per Sample: {usage['cpu_energy_joules'] / total:.5f} J/sample")

def main():
    # Run evaluation
    model, _ = get_pretrained_vit()

    if args.pruning_type == "structural":
        prune_by_importance(model, args.pruning_ratio)
    elif args.pruning_type == "unstructural":
        prune_by_masking(model, args.pruning_ratio)
    
    model.to(device)
    testloader = get_test_loader()
    warm_up_processors(model, (3, 224, 224), device)
    metrics = MultiClassClassificationMetrics()
    evaluate(model, testloader, device, metrics)

if __name__ == "__main__":
    main()
