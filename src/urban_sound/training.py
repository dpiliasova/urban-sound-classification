"""Reproducible two-stage fine-tuning with Mixup and early stopping."""

from __future__ import annotations

import random
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from urban_sound.config import TrainingConfig
from urban_sound.model import UrbanSoundDenseNet


@dataclass(frozen=True)
class EpochMetrics:
    stage: str
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    targets: list[int]
    predictions: list[int]
    probabilities: list[list[float]]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def mixup_batch(
    inputs: torch.Tensor, targets: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    mixing_weight = float(np.random.beta(alpha, alpha))
    mixing_weight = max(mixing_weight, 1.0 - mixing_weight)
    permutation = torch.randperm(inputs.shape[0], device=inputs.device)
    mixed = mixing_weight * inputs + (1.0 - mixing_weight) * inputs[permutation]
    return mixed, targets, targets[permutation], mixing_weight


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    weighted_correct = 0.0
    sample_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mixed, targets_a, targets_b, weight = mixup_batch(inputs, targets, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(mixed)
            loss = weight * criterion(logits, targets_a) + (1.0 - weight) * criterion(
                logits, targets_b
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)
        batch_size = inputs.shape[0]
        total_loss += float(loss.item()) * batch_size
        weighted_correct += weight * int((predictions == targets_a).sum().item())
        weighted_correct += (1.0 - weight) * int((predictions == targets_b).sum().item())
        sample_count += batch_size

    return total_loss / sample_count, weighted_correct / sample_count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[list[float]] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            probabilities = torch.softmax(logits.float(), dim=1)

            total_loss += float(loss.item()) * inputs.shape[0]
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(probabilities.argmax(dim=1).cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())

    correct = sum(a == b for a, b in zip(targets_all, predictions_all, strict=True))
    sample_count = len(targets_all)
    return EvaluationResult(
        loss=total_loss / sample_count,
        accuracy=correct / sample_count,
        targets=targets_all,
        predictions=predictions_all,
        probabilities=probabilities_all,
    )


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _update_best(
    model: nn.Module,
    validation_accuracy: float,
    best_accuracy: float,
    best_state: dict[str, torch.Tensor] | None,
) -> tuple[float, dict[str, torch.Tensor] | None, bool]:
    if validation_accuracy <= best_accuracy:
        return best_accuracy, best_state, False
    return validation_accuracy, _snapshot(model), True


def fit_two_stage(
    model: UrbanSoundDenseNet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[list[EpochMetrics], float]:
    """Fit the head, then the full network; restore the best validation checkpoint."""
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history: list[EpochMetrics] = []
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    model.freeze_backbone()
    head_optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.head_learning_rate,
        weight_decay=config.weight_decay,
    )
    for epoch in range(1, config.head_epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            head_optimizer,
            device,
            config.head_mixup_alpha,
            scaler,
            amp_enabled,
        )
        validation = evaluate(model, validation_loader, criterion, device, amp_enabled)
        history.append(
            EpochMetrics(
                "head",
                epoch,
                train_loss,
                train_accuracy,
                validation.loss,
                validation.accuracy,
            )
        )
        best_accuracy, best_state, _ = _update_best(
            model, validation.accuracy, best_accuracy, best_state
        )
        print(
            f"[head {epoch:02d}/{config.head_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation.loss:.4f} val_acc={validation.accuracy:.4f}",
            flush=True,
        )

    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.finetune_learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    epochs_without_improvement = 0

    for epoch in range(1, config.finetune_epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.finetune_mixup_alpha,
            scaler,
            amp_enabled,
        )
        validation = evaluate(model, validation_loader, criterion, device, amp_enabled)
        scheduler.step()
        history.append(
            EpochMetrics(
                "finetune",
                epoch,
                train_loss,
                train_accuracy,
                validation.loss,
                validation.accuracy,
            )
        )
        best_accuracy, best_state, improved = _update_best(
            model, validation.accuracy, best_accuracy, best_state
        )
        print(
            f"[finetune {epoch:02d}/{config.finetune_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation.loss:.4f} val_acc={validation.accuracy:.4f} "
            f"best={best_accuracy:.4f}",
            flush=True,
        )
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(deepcopy(best_state))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    return history, best_accuracy


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
