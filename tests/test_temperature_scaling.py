"""Tests for accusleepy/temperature_scaling.py"""

import torch

from accusleepy.constants import IMAGE_HEIGHT
from accusleepy.models import SSANN
from accusleepy.temperature_scaling import ModelWithTemperature


class TestModelWithTemperature:
    """Tests for ModelWithTemperature wrapper"""

    def test_init(self):
        """Wrapper instantiates correctly"""
        base_model = SSANN(n_classes=3)
        model = ModelWithTemperature(base_model)

        # Check model is wrapped
        assert hasattr(model, "model")
        assert model.model is base_model
        # Check temperature parameter exists and is initialized to 1.5
        assert hasattr(model, "temperature")
        assert model.temperature.item() == 1.5
        # Temperature should be a trainable parameter
        assert model.temperature.requires_grad

    def test_temperature_scale(self):
        """Logits scaled by temperature"""
        base_model = SSANN(n_classes=3)
        model = ModelWithTemperature(base_model)

        # Set a specific temperature for testing
        temperature = 2.0
        model.temperature.data = torch.tensor([temperature])

        # Create sample logits
        batch_size = 4
        n_classes = 3
        logits = torch.randn(batch_size, n_classes)

        scaled_logits = model.temperature_scale(logits)

        # Scaled logits should be logits / temperature
        expected = logits / temperature
        torch.testing.assert_close(scaled_logits, expected)

    def test_forward(self):
        """Forward pass applies temperature scaling"""
        n_classes = 3
        batch_size = 4
        epochs_per_img = 9

        base_model = SSANN(n_classes=n_classes)
        model = ModelWithTemperature(base_model)
        model.eval()

        # Create input tensor
        x = torch.randn(batch_size, 1, IMAGE_HEIGHT, epochs_per_img)

        with torch.no_grad():
            output = model(x)

        # Output shape should match n_classes
        assert output.shape == (batch_size, n_classes)

        # Verify that temperature scaling is applied
        # Get raw logits from base model
        with torch.no_grad():
            raw_logits = base_model(x)
            expected = raw_logits / model.temperature.item()

        torch.testing.assert_close(output, expected)
