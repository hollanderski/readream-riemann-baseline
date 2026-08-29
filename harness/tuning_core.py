"""Optimizer, scheduler and training loop, COPIED VERBATIM from tuning_p10_v3.py.

tuning_p10_v3.py cannot be imported: it does `np.load("../PREPROCESSED_DATA/...")`
at module scope (line 24). bridge_nn.py hit the same wall and resolved it the same
way, copying the loop verbatim rather than reimplementing it. This file follows that
precedent.

Provenance, block by block:
  build_optimizer / build_scheduler : tuning_p10_v3.py lines ~280-360, verbatim,
        including the clamps (lr to [1e-8,1], weight_decay to [0,1], max_lr forced
        above lr, div_factor capped) and the try/except that continues without a
        scheduler on failure.
  train()  : tuning_p10_v3.py lines 365-624. Mechanics kept exactly: patience,
        min_delta, best-state restore, grad clip max_norm=1.0, OneCycleLR stepped
        PER BATCH. wandb logging and the GPU-memory checkpoint bail-out are removed
        (logging and infrastructure, not mechanics).

ONE DELIBERATE DIVERGENCE, and it is already the established convention in this
codebase rather than my invention: the loss is nn.NLLLoss(), not p10_v3's
nn.CrossEntropyLoss(). Both architectures end in LogSoftmax, so CrossEntropyLoss
double-counts the softmax. tuning_p12_stable.py line 246 is the corrected variant,
and bridge_nn.py documents the same call: "loss: NLLLoss -- her model ends in
LogSoftmax (p10_v3 uses CrossEntropyLoss there, which double-counts the softmax;
p12_stable is the corrected one)".
"""
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler


def build_optimizer(model, config):
    """VERBATIM from tuning_p10_v3.py, optimizer block."""
    optimizer_name = config.get("optimizer", "adam").lower()
    lr = max(1e-8, min(1.0, config["lr"]))                 # Clamp learning rate
    weight_decay = max(0, min(1.0, config["weight_decay"]))  # Clamp weight decay
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=config.get("momentum", 0.9),
                               weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
                                 eps=config.get("eps", 1e-8), weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr,
                            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
                            eps=config.get("eps", 1e-8), weight_decay=weight_decay)


def build_scheduler(optimizer, config, steps_per_epoch):
    """VERBATIM from tuning_p10_v3.py, scheduler block. Default is 'cycle'
    (OneCycleLR), which her comment records as the best performing."""
    lr = max(1e-8, min(1.0, config["lr"]))
    try:
        max_lr = max(lr * 1.1, config.get("max_lr", lr * 10))   # Ensure max_lr > lr
        name = config.get("scheduler", None)
        if name == "step":
            return lr_scheduler.StepLR(optimizer, step_size=config.get("step_size", 10),
                                       gamma=config.get("gamma", 0.1))
        if name == "cosine":
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        if name == "plateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=config.get("factor", 0.1),
                patience=max(1, config.get("scheduler_patience", 3)))
        if name == "cycle":
            pct_start = max(0.01, min(0.99, config.get("pct_start", 0.1)))
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, epochs=config["epochs"],
                steps_per_epoch=steps_per_epoch, pct_start=pct_start,
                three_phase=config.get("three_phase", True))
    except Exception as e:
        print(f"Error creating scheduler (continuing without): {e}")
    return None


def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, config,
          scheduler=None, device="cuda"):
    """Mechanics VERBATIM from tuning_p10_v3.py train(). Returns best val accuracy;
    the model is left holding the best state, as hers does."""
    epochs = config["epochs"]
    patience = config.get("patience", 10)
    min_delta = config.get("min_delta", 0.001)
    best_val_acc = 0
    best_model_state = None
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X.float())
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # CycleLR scheduler steps after each BATCH, hers does the same
            if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
                try:
                    scheduler.step()
                except Exception as e:
                    print(f"Scheduler step error: {e}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X.float()).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        val_acc = correct / max(total, 1)

        if scheduler is not None and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        elif scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        if val_acc > (best_val_acc + min_delta):
            best_val_acc = val_acc
            no_improve_count = 0
            best_model_state = {k: v.cpu().clone().detach()
                                for k, v in model.state_dict().items()}
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return best_val_acc


def test(model, test_dataloader, device="cuda"):
    """Predictions from the best-state model. Shape of hers, without wandb."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_dataloader:
            preds.append(model(X.to(device).float()).argmax(dim=1).cpu())
            trues.append(y)
    import torch as _t
    return _t.cat(preds).numpy(), _t.cat(trues).numpy()
