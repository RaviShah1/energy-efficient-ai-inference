import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

from data.dataset import get_train_val_loaders, get_test_loader
from models.pretrained import get_pretrained_vit
from models.pruning.structural_pruning import prune_by_importance

from models.train_utils.loss import get_ce_loss, get_distillation_loss
from models.train_utils.train_epoch import train_one_epoch, distill_one_epoch
from models.train_utils.evaluate_epoch import evaluate_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=["train", "distill"], help="Training mode")
parser.add_argument("--pruning_ratio", type=float, default=0.1, help="Student pruning ratio (0.0–1.0)")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
parser.add_argument("--save_prefix", type=str, default="weights", help="Prefix for saving model weights each epoch")
parser.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
parser.add_argument("--alpha", type=float, default=0.25, help="Weight for distillation loss, (1-alpha) for CE loss")
parser.add_argument("--scheduler", action="store_true", help="True to use cosine lr scheduler")
args = parser.parse_args()


def train(model_student, model_teacher, trainloader, valloader, testloader, optimizer, loss_fn, scheduler=None, epochs=10):
    evaluate_epoch(model_student, valloader, device, "Pre-Val")
    evaluate_epoch(model_student, testloader, device, "Pre-Test")
    
    for e in range(epochs):
        print(f"\n--- Epoch {e+1}---")

        if args.mode == "distill":
            avg_loss = distill_one_epoch(
                model_student,
                model_teacher,
                trainloader,
                loss_fn,
                optimizer,
                device,
                scheduler=scheduler
            )
        else:
            avg_loss = train_one_epoch(
                model_student,
                trainloader,
                loss_fn,
                optimizer,
                device,
                scheduler=scheduler
            )

        print(f"Epoch {e+1} Training Loss: {avg_loss:.4f}")
        evaluate_epoch(model_student, valloader, device, f"Val: epoch {e}")
        evaluate_epoch(model_student, testloader, device, f"Test: epoch {e}")

        # Save model weights
        torch.save(model_student.state_dict(), f"{args.save_prefix}_{e+1}.pth")
        print(f"Saved model to: {args.save_prefix}_{e+1}.pth")


def main():
    # Load teacher model if distillation mode
    model_teacher = None
    if args.mode == "distill":
        model_teacher, _ = get_pretrained_vit()
        model_teacher.to(device)
        model_teacher.eval()

    # Load student model (copy and prune)
    model_student, _ = get_pretrained_vit()
    prune_by_importance(model_student, ratio=args.pruning_ratio, include_attn=True)
    model_student.to(device)

    # Dataloaders
    trainloader, valloader = get_train_val_loaders(batch_size=args.batch_size)
    testloader = get_test_loader(batch_size=args.batch_size)

    # Optimizer
    optimizer = optim.AdamW(model_student.parameters(), lr=args.lr, weight_decay=0.01)

    # Scheduler
    scheduler = None
    if args.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    if args.mode == "distill":
        loss_fn = get_distillation_loss(temperature=args.temperature, alpha=args.alpha)
    else:
        loss_fn = get_ce_loss()

    # Run training
    train(
        model_student,
        model_teacher,
        trainloader,
        valloader,
        testloader,
        optimizer,
        loss_fn,
        scheduler=scheduler,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
