import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from urban_sound.config import AudioConfig
from urban_sound.data import UrbanSoundDataset, make_cache


def test_dataset_extracts_caches_and_preloads_a_clip(tmp_path):
    audio_root = tmp_path / "audio"
    fold_dir = audio_root / "fold1"
    fold_dir.mkdir(parents=True)
    audio_path = fold_dir / "sample.wav"
    sample_rate = 8_000
    timeline = np.arange(sample_rate // 10) / sample_rate
    sf.write(audio_path, np.sin(2 * np.pi * 440 * timeline), sample_rate)

    metadata = pd.DataFrame(
        [
            {
                "slice_file_name": audio_path.name,
                "fold": 1,
                "classID": 3,
                "class": "dog_bark",
            }
        ]
    )
    config = AudioConfig(
        sample_rate=sample_rate,
        duration_seconds=0.1,
        n_mels=16,
        n_fft=128,
        hop_length=32,
    )
    cache = make_cache(tmp_path / "cache", config, keep_in_memory=True)
    dataset = UrbanSoundDataset(metadata, audio_root, cache, preload=True)

    spectrogram, label = dataset[0]
    assert spectrogram.shape[0:2] == (3, 16)
    assert spectrogram.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert label == 3
    assert cache.in_memory_items == 1
    assert len(list((tmp_path / "cache").glob("*.pt"))) == 1
