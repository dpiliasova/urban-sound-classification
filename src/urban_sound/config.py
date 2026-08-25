"""Typed experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22_050
    duration_seconds: float = 4.0
    n_mels: int = 128
    n_fft: int = 2_048
    hop_length: int = 512

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.duration_seconds)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1_337
    batch_size: int = 64
    head_epochs: int = 9
    finetune_epochs: int = 50
    head_learning_rate: float = 1e-3
    finetune_learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    head_mixup_alpha: float = 0.3
    finetune_mixup_alpha: float = 0.2
    patience: int = 15
    num_workers: int = 4
    use_amp: bool = True


def save_config(path: Path, audio: AudioConfig, training: TrainingConfig) -> None:
    """Persist the exact configuration used for an experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"audio": asdict(audio), "training": asdict(training)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
