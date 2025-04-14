import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from data.dataset import get_train_val_loaders, get_test_loader
from models.pretrained import get_pretrained_vit
from models.pruning.structural_pruning import prune_by_importance

from models.train_utils.loss import get_ce_loss
from models.train_utils.train_epoch import train_one_epoch
from models.train_utils.evaluate_epoch import evaluate_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--pruning_ratio", type=float, default=0.1, help="Student pruning ratio (0.0–1.0)")
parser.add_argument("--epochs", type=int, default=10, help="Number of distillation epochs")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
args = parser.parse_args()

def train(model_student, model_teacher, trainloader, valloader, testloader, optimizer, loss_fn, scheduler=None, epochs=10):
    for e in range(epochs):
        print(f"\n--- Epoch {e+1}---")
        avg_loss = train_one_epoch(model_student, trainloader, loss_fn, optimizer, device, scheduler)
        print(f"Epoch {e+1} Training Loss: {avg_loss:.4f}")

        evaluate_epoch(model_student, valloader, device, f"Val: epoch {e}")
        evaluate_epoch(model_student, testloader, device, f"Test: epoch {e}")



def main():
    # Load teacher model
    model_teacher, _ = get_pretrained_vit()
    model_teacher.to(device)
    model_teacher.eval()

    # Load student model (copy and prune)
    model_student, _ = get_pretrained_vit()
    prune_by_importance(model_student, ratio=args.pruning_ratio)
    model_student.to(device)

    # Dataloaders
    trainloader, valloader = get_train_val_loaders(batch_size=args.batch_size)
    testloader = get_test_loader(batch_size=args.batch_size)

    # Optimizer
    optimizer = optim.AdamW(model_student.parameters(), lr=args.lr)

    # Loss
    loss_fn = get_ce_loss()

    # Run distillation
    train(
        model_student,
        model_teacher,
        trainloader,
        valloader,
        testloader,
        optimizer,
        loss_fn,
        epochs=args.epochs,
    )

if __name__ == "__main__":
    main()
