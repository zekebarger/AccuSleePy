"""Tests for accusleepy/classification.py"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import accusleepy.constants as c
from accusleepy.classification import AccuSleepImageDataset


class TestAccuSleepImageDataset:
    """Tests for AccuSleepImageDataset"""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset with images and annotations"""
        # Create sample images
        n_images = 5
        img_height = 30
        img_width = 9
        filenames = []
        labels = []

        for i in range(n_images):
            filename = f"image_{i}.png"
            filenames.append(filename)
            labels.append(i % 3)  # Cycle through 0, 1, 2

            # Create and save a small test image
            img = np.random.randint(0, 256, (img_height, img_width), dtype=np.uint8)
            Image.fromarray(img).save(tmp_path / filename)

        # Create annotations file
        annotations_df = pd.DataFrame({c.FILENAME_COL: filenames, c.LABEL_COL: labels})
        annotations_file = tmp_path / "annotations.csv"
        annotations_df.to_csv(annotations_file, index=False)

        return tmp_path, annotations_file, n_images, labels

    def test_len(self, sample_dataset):
        """Returns correct length"""
        img_dir, annotations_file, n_images, _ = sample_dataset

        dataset = AccuSleepImageDataset(
            annotations_file=str(annotations_file), img_dir=str(img_dir)
        )

        assert len(dataset) == n_images

    def test_getitem(self, sample_dataset):
        """Returns image and label tuple"""
        img_dir, annotations_file, _, labels = sample_dataset

        dataset = AccuSleepImageDataset(
            annotations_file=str(annotations_file), img_dir=str(img_dir)
        )

        # Get first item
        image, label = dataset[0]

        # Image should be a tensor with shape (channels, height, width)
        assert image.ndim == 3
        assert image.shape[0] == 1  # Grayscale = 1 channel
        # Label should match expected
        assert label == labels[0]
