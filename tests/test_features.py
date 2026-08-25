import pytest
import torch

from urban_sound.features import mask_axis, normalize_spectrogram


def test_normalization_centres_a_spectrogram():
    spectrogram = torch.linspace(-80, 0, steps=24).reshape(1, 4, 6)
    normalized = normalize_spectrogram(spectrogram)
    assert normalized.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert normalized.std().item() == pytest.approx(1.0, abs=1e-6)


def test_mask_uses_normalized_mean_value():
    generator = torch.Generator().manual_seed(4)
    normalized = torch.ones((1, 8, 10))
    masked = mask_axis(normalized.clone(), axis=-1, maximum_width=8, generator=generator)
    assert torch.any(masked == 0)
    assert set(masked.unique().tolist()).issubset({0.0, 1.0})


def test_mask_does_not_change_shape():
    generator = torch.Generator().manual_seed(7)
    spectrogram = torch.randn((1, 128, 173))
    masked = mask_axis(spectrogram, axis=-2, maximum_width=30, generator=generator)
    assert masked.shape == (1, 128, 173)
