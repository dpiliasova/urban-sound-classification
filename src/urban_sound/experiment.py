"""End-to-end development-split and rotating-fold experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from urban_sound.config import AudioConfig, TrainingConfig, save_config
from urban_sound.data import CLASS_NAMES, UrbanSoundDataset, make_cache, read_metadata
from urban_sound.model import UrbanSoundDenseNet
from urban_sound.splits import limit_split, next_validation_fold, split_metadata
from urban_sound.training import evaluate, fit_two_stage, seed_worker, set_reproducible_seed


def _make_loader(
    dataset: UrbanSoundDataset,
    config: TrainingConfig,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#182230",
                fontsize=8,
            )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_fold_plot(accuracies: list[float], output_path: Path) -> None:
    folds = np.arange(1, len(accuracies) + 1)
    mean_accuracy = float(np.mean(accuracies))
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(folds, np.multiply(accuracies, 100), color="#277da1")
    axis.axhline(
        mean_accuracy * 100,
        color="#f3722c",
        linestyle="--",
        label=f"Mean: {mean_accuracy:.2%}",
    )
    axis.set(
        xlabel="Official test fold",
        ylabel="Accuracy, %",
        title="UrbanSound8K accuracy by official fold",
        xticks=folds,
        ylim=(0, 100),
    )
    axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_fold(
    data_root: Path,
    output_dir: Path,
    test_fold: int,
    val_fold: int,
    audio_config: AudioConfig,
    training_config: TrainingConfig,
    pretrained: bool = True,
    max_samples_per_split: int | None = None,
) -> dict[str, object]:
    """Train on eight folds, select on one fold, and evaluate once on the test fold."""
    set_reproducible_seed(training_config.seed + test_fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "config.json", audio_config, training_config)

    metadata, audio_root = read_metadata(data_root)
    train_frame, validation_frame, test_frame = split_metadata(metadata, test_fold, val_fold)
    train_frame = limit_split(train_frame, max_samples_per_split)
    validation_frame = limit_split(validation_frame, max_samples_per_split)
    test_frame = limit_split(test_frame, max_samples_per_split)

    cache = make_cache(data_root, audio_config)
    train_dataset = UrbanSoundDataset(train_frame, audio_root, cache, augment=True)
    validation_dataset = UrbanSoundDataset(validation_frame, audio_root, cache, augment=False)
    test_dataset = UrbanSoundDataset(test_frame, audio_root, cache, augment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = _make_loader(train_dataset, training_config, True, device)
    validation_loader = _make_loader(validation_dataset, training_config, False, device)
    test_loader = _make_loader(test_dataset, training_config, False, device)

    model = UrbanSoundDenseNet(pretrained=pretrained).to(device)
    history, best_validation_accuracy = fit_two_stage(
        model,
        train_loader,
        validation_loader,
        training_config,
        device,
        output_dir / "best_model.pth",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
    test_result = evaluate(model, test_loader, criterion, device)
    report = classification_report(
        test_result.targets,
        test_result.predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_result.targets,
        test_result.predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    metrics: dict[str, object] = {
        "test_fold": test_fold,
        "validation_fold": val_fold,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "best_validation_accuracy": best_validation_accuracy,
        "test_accuracy": test_result.accuracy,
        "test_loss": test_result.loss,
        "classification_report": report,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(
        json.dumps([asdict(item) for item in history], indent=2), encoding="utf-8"
    )

    predictions = test_frame[["slice_file_name", "fold", "classID", "class"]].copy()
    predictions["predicted_class_id"] = test_result.predictions
    predictions["predicted_class"] = [CLASS_NAMES[value] for value in test_result.predictions]
    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions[f"probability_{class_name}"] = [
            values[class_id] for values in test_result.probabilities
        ]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    _save_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    return metrics


def run_cross_validation(
    data_root: Path,
    output_dir: Path,
    audio_config: AudioConfig,
    training_config: TrainingConfig,
    pretrained: bool = True,
    max_samples_per_split: int | None = None,
) -> dict[str, object]:
    """Evaluate each official fold once after validation-based checkpoint selection."""
    fold_results: list[dict[str, object]] = []
    for test_fold in range(1, 11):
        val_fold = next_validation_fold(test_fold)
        result = run_fold(
            data_root=data_root,
            output_dir=output_dir / f"fold_{test_fold}",
            test_fold=test_fold,
            val_fold=val_fold,
            audio_config=audio_config,
            training_config=training_config,
            pretrained=pretrained,
            max_samples_per_split=max_samples_per_split,
        )
        fold_results.append(result)

    accuracies = [float(result["test_accuracy"]) for result in fold_results]
    summary: dict[str, object] = {
        "fold_accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "standard_deviation": float(np.std(accuracies)),
        "fold_results": fold_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_fold_plot(accuracies, output_dir / "fold_accuracy.png")
    return summary
