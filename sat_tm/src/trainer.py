import time

import torch


def train(
    model, loaders, device, cfg,
):
    """Run training with AMP, grad accumulation, early stopping, LR scheduler."""
    import torch.nn as nn
    import torch.optim as optim

    NUM_EPOCHS = cfg["training"]["epochs"]
    LR = cfg["training"]["learning_rate"]
    USE_AMP = cfg["training"]["amp"] and torch.cuda.is_available()
    GRAD_ACCUM_STEPS = 1
    EARLY_STOP_PATIENCE = cfg["training"]["early_stop_patience"]
    USE_EARLY_STOPPING = cfg["training"].get("use_early_stopping", True)
    USE_SCHEDULER = cfg["training"].get("use_scheduler", True)

    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    scaler_amp = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    train_loss_history, val_loss_history = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    total_start = time.time()
    best_epoch = None
    best_epoch_time = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, num_samples = 0.0, 0
        optimizer.zero_grad()

        t0 = time.time()
        for i, (x, _) in enumerate(loaders["train"]):
            x = x.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                y = model(x)
                loss = criterion(y, x).mean() / GRAD_ACCUM_STEPS

            scaler_amp.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loaders["train"]):
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()

            running_loss += loss.item() * GRAD_ACCUM_STEPS * x.size(0)
            num_samples += x.size(0)

        train_epoch_loss = running_loss / num_samples
        train_loss_history.append(train_epoch_loss)

        # Validation
        model.eval()
        val_running, val_samples = 0.0, 0
        with torch.no_grad():
            for x, _ in loaders["val"]:
                x = x.to(device, non_blocking=True)
                y = model(x)
                sample_errors = criterion(y, x).mean(dim=(1, 2))
                val_running += sample_errors.sum().item()
                val_samples += x.size(0)

        val_epoch_loss = val_running / val_samples
        val_loss_history.append(val_epoch_loss)

        if USE_SCHEDULER:
            scheduler.step(val_epoch_loss)

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train_loss: {train_epoch_loss:.6f}"
              f" | val_loss: {val_epoch_loss:.6f} | {dt:.1f}s")

        if val_epoch_loss < best_val_loss - 1e-6:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            best_epoch_time = time.time() - total_start
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if USE_EARLY_STOPPING and epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no val improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    total_train_time = time.time() - total_start
    print(f"Total training time: {total_train_time:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    return (model, train_loss_history, val_loss_history,
            best_val_loss, best_epoch, best_epoch_time, total_train_time)
