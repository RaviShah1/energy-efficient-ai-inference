import torch
import time
from tqdm import tqdm

from energy.energy_tracker import EnergyTracker
from models.pretrained import get_pretrained_vit
from data.dataset import get_test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


# Evaluation function with energy tracking
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    energy = EnergyTracker()
    energy.start()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            inputs = images.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)

    usage = energy.end()
    energy.shutdown()

    total_time = time.time() - start_time
    avg_time_per_sample = total_time / total
    accuracy = correct / total

    print(f"\nAccuracy on CIFAR-100 test set: {accuracy:.4f}")
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Average Inference Time per Sample: {avg_time_per_sample:.5f} seconds")
    print(f"Total GPU Energy Consumption: {usage['gpu_energy_joules']:.2f} Joules")
    print(f"Average GPU Energy per Sample: {usage['gpu_energy_joules'] / total:.5f} J/sample")
    print(f"Total CPU Energy Consumption: {usage['cpu_energy_joules']:.2f} Joules")
    print(f"Average CPU Energy per Sample: {usage['cpu_energy_joules'] / total:.5f} J/sample")

def main():
    # Run evaluation
    model, _ = get_pretrained_vit()
    model.to(device)
    testloader = get_test_loader()
    evaluate(model, testloader, device)

if __name__ == "__main__":
    main()
