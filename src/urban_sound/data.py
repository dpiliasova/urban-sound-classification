"""Dataset discovery and PyTorch dataset implementation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from urban_sound.config import AudioConfig
from urban_sound.features import MelSpectrogramCache, augment_spectrogram, normalize_spectrogram
from urban_sound.splits import validate_metadata

CLASS_NAMES = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)


def resolve_dataset_layout(data_root: Path) -> tuple[Path, Path]:
    """Support the official archive and the common flattened Kaggle layout."""
    official_metadata = data_root / "metadata" / "UrbanSound8K.csv"
    if official_metadata.exists() and (data_root / "audio").is_dir():
        return official_metadata, data_root / "audio"

    flat_metadata = data_root / "UrbanSound8K.csv"
    if flat_metadata.exists() and (data_root / "fold1").is_dir():
        return flat_metadata, data_root

    raise FileNotFoundError(
        "Could not find UrbanSound8K. Expected metadata/UrbanSound8K.csv and audio/fold1, "
        "or UrbanSound8K.csv and fold1 directly under the data root."
    )


def read_metadata(data_root: Path) -> tuple[pd.DataFrame, Path]:
    metadata_path, audio_root = resolve_dataset_layout(data_root)
    metadata = pd.read_csv(metadata_path)
    validate_metadata(metadata)
    return metadata, audio_root


class UrbanSoundDataset(Dataset[tuple[torch.Tensor, int]]):
    """Return normalized three-channel log-mel spectrograms and class IDs."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        audio_root: Path,
        cache: MelSpectrogramCache,
        augment: bool = False,
    ) -> None:
        self.records = metadata.to_dict(orient="records")
        self.audio_root = audio_root
        self.cache = cache
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.records[index]
        audio_path = self.audio_root / f"fold{int(row['fold'])}" / str(row["slice_file_name"])
        spectrogram = normalize_spectrogram(self.cache.get(audio_path))
        if self.augment:
            spectrogram = augment_spectrogram(spectrogram)
        spectrogram = spectrogram.repeat(3, 1, 1)
        return spectrogram, int(row["classID"])


def make_cache(data_root: Path, audio_config: AudioConfig) -> MelSpectrogramCache:
    return MelSpectrogramCache(data_root.parent / "cache", audio_config)
