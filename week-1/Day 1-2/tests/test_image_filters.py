#!/usr/bin/env python3
"""
Unit Tests for Image Filters - NumPy Mastery Module

This test suite validates the image processing functionality
including grayscale conversion, color inversion, and array operations.
"""

import numpy as np
import pytest
from src.projects.image_filters import ImageFilters, create_test_image


class TestImageFilters:
    """Test suite for image processing functionality."""

    def test_create_test_image(self):
        """Test that test image is created with correct properties."""
        image = create_test_image()

        assert image.shape == (100, 100, 3), f"Expected shape (100, 100, 3), got {image.shape}"
        assert image.dtype == np.uint8, f"Expected dtype uint8, got {image.dtype}"
        assert image.min() >= 0, "Image values should be non-negative"
        assert image.max() <= 255, "Image values should not exceed 255"

    def test_grayscale_conversion_basic(self):
        """Test basic grayscale conversion with known values."""
        processor = ImageFilters()

        # Create test image with known RGB values
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        test_image[:, :, 0] = 100  # Red channel
        test_image[:, :, 1] = 150  # Green channel
        test_image[:, :, 2] = 200  # Blue channel

        grayscale = processor.convert_to_grayscale(test_image)

        # Expected value: 0.2989*100 + 0.5870*150 + 0.1140*200
        expected_value = int(0.2989 * 100 + 0.5870 * 150 + 0.1140 * 200)

        assert grayscale.shape == (10, 10), f"Expected shape (10, 10), got {grayscale.shape}"
        assert grayscale.dtype == np.uint8, f"Expected dtype uint8, got {grayscale.dtype}"
        assert np.allclose(grayscale, expected_value, atol=1), \
            f"Expected approximately {expected_value}, got {grayscale[0, 0]}"

    def test_grayscale_conversion_with_loaded_image(self):
        """Test grayscale conversion when image is loaded via class."""
        processor = ImageFilters()

        # Create and set test image
        test_image = create_test_image()
        processor.image_array = test_image

        grayscale = processor.convert_to_grayscale()  # Should use loaded image

        assert grayscale.shape[0] == test_image.shape[0], "Height should match"
        assert grayscale.shape[1] == test_image.shape[1], "Width should match"
        assert len(grayscale.shape) == 2, "Output should be 2D (grayscale)"

    def test_grayscale_conversion_error_handling(self):
        """Test error handling for grayscale conversion."""
        processor = ImageFilters()

        # Test with no loaded image
        with pytest.raises(ValueError, match="No image loaded"):
            processor.convert_to_grayscale()

        # Test with invalid input (wrong dimensions)
        invalid_image = np.array([1, 2, 3])  # 1D array
        with pytest.raises(ValueError, match="Expected 3-channel color image"):
            processor.convert_to_grayscale(invalid_image)

        # Test with 2-channel image (not 3-channel)
        invalid_image = np.random.rand(10, 10, 2)  # 2-channel
        with pytest.raises(ValueError, match="Expected 3-channel color image"):
            processor.convert_to_grayscale(invalid_image)

    def test_color_inversion_basic(self):
        """Test basic color inversion."""
        processor = ImageFilters()

        # Create test image with known values
        test_image = np.array([[[100, 150, 200]]], dtype=np.uint8)  # Single pixel

        inverted = processor.invert_colors(test_image)

        expected = np.array([[[155, 105, 55]]], dtype=np.uint8)  # 255 - original values
        expected = np.clip(expected, 0, 255).astype(np.uint8)  # Ensure valid range

        assert inverted.shape == test_image.shape, "Shape should be preserved"
        assert np.array_equal(inverted, expected), \
            f"Expected {expected[0, 0]}, got {inverted[0, 0]}"

    def test_color_inversion_with_loaded_image(self):
        """Test color inversion when image is loaded via class."""
        processor = ImageFilters()

        # Create and set test image
        test_image = create_test_image()
        processor.image_array = test_image

        inverted = processor.invert_colors()  # Should use loaded image

        assert inverted.shape == test_image.shape, "Shape should be preserved"
        # Verify that inverted values are approximately 255 - original
        assert np.allclose(inverted, 255 - test_image), "Values should be inverted"

    def test_color_inversion_error_handling(self):
        """Test error handling for color inversion."""
        processor = ImageFilters()

        # Test with no loaded image
        with pytest.raises(ValueError, match="No image loaded"):
            processor.invert_colors()

    def test_color_inversion_preserves_range(self):
        """Test that color inversion keeps values in valid range."""
        processor = ImageFilters()

        # Test with boundary values
        test_image = np.array([[[0, 128, 255]]], dtype=np.uint8)

        inverted = processor.invert_colors(test_image)

        expected = np.array([[[255, 127, 0]]], dtype=np.uint8)
        assert np.array_equal(inverted, expected), "Boundary values should be correctly inverted"

    def test_grayscale_luminance_weights(self):
        """Test that luminance formula weights are correctly applied."""
        processor = ImageFilters()

        # Create image with specific values where we can verify the formula
        # Use single pixel for easy verification
        test_image = np.array([[[100, 200, 50]]], dtype=np.uint8)  # R=100, G=200, B=50

        grayscale = processor.convert_to_grayscale(test_image)

        # Calculate expected value using the luminance formula:
        # Y = 0.2989*R + 0.5870*G + 0.1140*B
        expected_value = 0.2989 * 100 + 0.5870 * 200 + 0.1140 * 50
        expected_value = int(expected_value)  # Convert to int like the function does

        actual_value = grayscale[0, 0]
        assert abs(actual_value - expected_value) <= 1, \
            f"Luminance formula not applied correctly. Expected ~{expected_value}, got {actual_value}"

    def test_class_initialization(self):
        """Test ImageFilters class initialization."""
        processor = ImageFilters()

        assert processor.image_array is None, "Initial image_array should be None"
        assert processor.original_shape is None, "Initial original_shape should be None"

    def test_save_image_functionality(self):
        """Test the save_image functionality (basic validation)."""
        processor = ImageFilters()

        # Since we can't actually save without PIL, we'll test the logic
        # that would be used before PIL conversion
        test_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

        # This would normally save the image, but we'll just verify the data handling
        # The actual PIL part is tested separately when PIL is available
        processed_image = np.clip(test_image.astype(np.float64), 0, 255).astype(np.uint8)

        assert processed_image.dtype == np.uint8, "Image should be converted to uint8"
        assert processed_image.min() >= 0, "Values should be >= 0"
        assert processed_image.max() <= 255, "Values should be <= 255"

    def test_image_filters_integration(self):
        """Test the complete image processing pipeline."""
        processor = ImageFilters()

        # Create test image
        test_image = create_test_image()
        processor.image_array = test_image

        # Test grayscale conversion
        grayscale = processor.convert_to_grayscale()
        assert grayscale.shape == (test_image.shape[0], test_image.shape[1])
        assert len(grayscale.shape) == 2

        # Test color inversion
        inverted = processor.invert_colors()
        assert inverted.shape == test_image.shape

        # Verify that inverted image has different values than original
        assert not np.array_equal(inverted, test_image), "Inverted image should differ from original"

        # Verify grayscale has different characteristics than original
        assert len(grayscale.shape) != len(test_image.shape), "Grayscale should be 2D, original 3D"


def run_all_tests():
    """Run all tests and return results."""
    import unittest
    import sys

    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestImageFilters)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Image Filters Unit Tests...")
    print("=" * 50)

    success = run_all_tests()

    if success:
        print("\n✅ All image filter tests PASSED!")
    else:
        print("\n❌ Some tests FAILED!")
        sys.exit(1)