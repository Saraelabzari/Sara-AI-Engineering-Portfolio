import pytest
import torch
from PIL import Image

from src.classifier import build_model, build_transforms, run_epoch, save_checkpoint


def test_build_transforms_rejects_too_small_size():
    with pytest.raises(ValueError, match="at least 32"):
        build_transforms(16)


def test_build_transforms_produces_expected_tensor_shape():
    _, eval_transform = build_transforms(64)
    tensor = eval_transform(Image.new("RGB", (40, 30), color="white"))
    assert tuple(tensor.shape) == (3, 64, 64)


def test_build_model_rejects_one_class():
    with pytest.raises(ValueError, match="at least 2"):
        build_model(1, pretrained=False)


def test_build_model_replaces_classifier_head():
    model = build_model(3, pretrained=False)
    assert model.fc.out_features == 3
    assert all(parameter.requires_grad for parameter in model.fc.parameters())


def test_save_checkpoint_writes_metadata(tmp_path):
    model = build_model(2, pretrained=False)
    destination = tmp_path / "nested" / "model.pth"
    save_checkpoint(destination, model, ["cat", "dog"], image_size=128)
    checkpoint = torch.load(destination, map_location="cpu", weights_only=True)
    assert checkpoint["class_names"] == ["cat", "dog"]
    assert checkpoint["image_size"] == 128
    assert checkpoint["architecture"] == "resnet18"


def test_run_epoch_returns_loss_and_accuracy():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    inputs = torch.rand(4, 3, 2, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels), batch_size=2
    )
    loss, accuracy = run_epoch(
        model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert loss >= 0
    assert 0 <= accuracy <= 1
