from tqdm import tqdm

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, scheduler=None, log_interval=50):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Batch {batch_idx+1}/{num_batches}] Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

    avg_loss = total_loss / num_batches
    return avg_loss
