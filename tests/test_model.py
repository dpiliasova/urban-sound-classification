import torch

from urban_sound.model import UrbanSoundDenseNet


def test_model_returns_ten_logits_per_clip():
    model = UrbanSoundDenseNet(pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 128, 173))
    assert logits.shape == (2, 10)


def test_freeze_and_unfreeze_backbone():
    model = UrbanSoundDenseNet(pretrained=False)
    model.freeze_backbone()
    assert not any(parameter.requires_grad for parameter in model.backbone.features.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.classifier.parameters())
    model.unfreeze_backbone()
    assert all(parameter.requires_grad for parameter in model.parameters())
