
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from training import train_epoch, evaluate, save_best_model


def train_basic_classifier(model, train_loader, val_loader, config):

    NUM_EPOCHS = config["num_epochs"]
    LEARNING_RATE = config["lr"]
    PATIENCE = config["patience"]
    DEVICE = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Best Top5 Accuracy
    best_val_acc = 0
    best_model_state = None
    epochs_without_improvement = 0
        
    history = {
        'train': [],
        'val': []
    }

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
        
    # Training loop
    for epoch in range(NUM_EPOCHS):

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        # Learning rate scheduling
        scheduler.step()
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train - Loss: {train_metrics['loss_avg']:.4f}, Rank Loss: {train_metrics['loss_rank_avg']:.4f}")
        print(f"          Top-1: {train_metrics['top1_acc']:.4f}, Top-5: {train_metrics['top5_acc']:.4f}")
        print(f"          Rank Top-1: {train_metrics['top1_rank_acc']:.4f}, Rank Top-5: {train_metrics['top5_rank_acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss_avg']:.4f}, Rank Loss: {val_metrics['loss_rank_avg']:.4f}")
        print(f"          Top-1: {val_metrics['top1_acc']:.4f}, Top-5: {val_metrics['top5_acc']:.4f}")
        print(f"          Rank Top-1: {val_metrics['top1_rank_acc']:.4f}, Rank Top-5: {val_metrics['top5_rank_acc']:.4f}")
        
        best_val_acc, best_model_state, improved = save_best_model(
            val_metrics["top5_acc"], best_val_acc, model, best_model_state
        )

        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    return best_val_acc, best_model_state, history