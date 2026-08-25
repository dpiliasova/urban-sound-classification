"""Audio loading, log-mel extraction, caching, and spectrogram augmentation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from hashlib import sha1
from pathlib import Path

import librosa
import numpy as np
import torch

from urban_sound.config import AudioConfig


def normalize_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    """Normalize one spectrogram before masking, making zero the sample mean."""
    mean = spectrogram.mean()
    standard_deviation = spectrogram.std().clamp_min(1e-6)
    return (spectrogram - mean) / standard_deviation


def mask_axis(
    spectrogram: torch.Tensor,
    axis: int,
    maximum_width: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mask a random frequency or time interval with the normalized mean."""
    if maximum_width <= 0:
        return spectrogram
    axis_size = spectrogram.shape[axis]
    width_limit = min(maximum_width, axis_size)
    width = int(torch.randint(0, width_limit + 1, (1,), generator=generator).item())
    if width == 0:
        return spectrogram
    start = int(torch.randint(0, axis_size - width + 1, (1,), generator=generator).item())
    index = [slice(None)] * spectrogram.ndim
    index[axis] = slice(start, start + width)
    spectrogram[tuple(index)] = 0.0
    return spectrogram


def augment_spectrogram(
    spectrogram: torch.Tensor,
    generator: torch.Generator | None = None,
    frequency_masks: int = 3,
    time_masks: int = 3,
) -> torch.Tensor:
    """Apply lightweight augmentation after per-example normalization."""
    augmented = spectrogram.clone()

    if torch.rand((), generator=generator).item() < 0.5:
        shift = int(torch.randint(-5, 6, (1,), generator=generator).item())
        augmented = torch.roll(augmented, shifts=shift, dims=-1)

    if torch.rand((), generator=generator).item() < 0.5:
        noise_scale = 0.005 + 0.015 * torch.rand((), generator=generator).item()
        noise = torch.randn(augmented.shape, generator=generator, dtype=augmented.dtype)
        augmented = augmented + noise_scale * noise

    for _ in range(frequency_masks):
        if torch.rand((), generator=generator).item() < 0.5:
            augmented = mask_axis(augmented, axis=-2, maximum_width=30, generator=generator)
    for _ in range(time_masks):
        if torch.rand((), generator=generator).item() < 0.5:
            augmented = mask_axis(augmented, axis=-1, maximum_width=40, generator=generator)
    return augmented


class MelSpectrogramCache:
    """Extract each log-mel spectrogram once and reuse it across fold runs."""

    def __init__(self, cache_dir: Path, config: AudioConfig) -> None:
        self.cache_dir = cache_dir
        self.config = config
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, audio_path: Path) -> Path:
        identity = {
            "file": f"{audio_path.parent.name}/{audio_path.name}",
            "audio": asdict(self.config),
        }
        digest = sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_dir / f"{audio_path.stem}-{digest}.pt"

    def get(self, audio_path: Path) -> torch.Tensor:
        cache_path = self._cache_path(audio_path)
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        spectrogram = self._extract(audio_path)
        temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(spectrogram, temporary_path)
        os.replace(temporary_path, cache_path)
        return spectrogram

    def _extract(self, audio_path: Path) -> torch.Tensor:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio clip not found: {audio_path}")

        waveform, _ = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        waveform = librosa.util.fix_length(waveform, size=self.config.target_samples)
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
        return torch.from_numpy(log_mel).to(dtype=torch.float32).unsqueeze(0)
