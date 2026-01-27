#!/usr/bin/env python3
"""
Image Processing Mini-Project - NumPy Mastery

This module demonstrates that images are 3D NumPy arrays (Height × Width × Channels)
and that filters are mathematical operations applied to these arrays.

KEY CONCEPT: An image is just a 3D array of numbers!
- Height: Number of rows
- Width: Number of columns
- Channels: Color channels (RGB = 3, RGBA = 4)

CONSTITUTION COMPLIANCE:
- ✓ Vectorization over loops (where applicable)
- ✓ Clear variable names (image_array, grayscale_image, etc.)
- ✓ Proper comments explaining WHY operations are used
- ✓ Emphasis on images as 3D arrays
"""

import numpy as np
import os


# Import PIL only if available (for image loading/saving)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Import matplotlib only if available (for display functions)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class ImageFilters:
    """
    Image processing class demonstrating NumPy array operations on images.

    Emphasizes that images are 3D NumPy arrays (Height × Width × Channels)
    and filters are mathematical operations on these arrays.
    """

    def __init__(self):
        """
        Initialize the ImageFilters class.
        """
        self.image_array = None
        self.original_shape = None
        print("ImageFilters initialized. Ready to process images!")
        print("Remember: Images are 3D arrays (Height × Width × Channels)")

    def load_image(self, image_path):
        """
        Load an image and convert to NumPy array.

        This method emphasizes the core concept that images are just 3D arrays
        of numbers that can be manipulated with mathematical operations.

        Args:
            image_path (str): Path to image file (JPG, PNG, etc.)

        Returns:
            np.ndarray: Image as 3D array with shape (Height, Width, Channels)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ImportError: If PIL is not available
            ValueError: If image cannot be loaded
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) not available. Cannot load images.")

        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image using PIL and convert to RGB to ensure 3-channel format
        # WHY: This handles different image formats (RGBA, grayscale, etc.)
        pil_image = Image.open(image_path)

        # Convert to RGB to ensure consistent 3-channel format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert PIL image to NumPy array - this is the key transformation!
        # NOW the image is just a 3D array of numbers!
        self.image_array = np.array(pil_image)
        self.original_shape = self.image_array.shape

        print(f"Image loaded successfully!")
        print(f"Shape: {self.image_array.shape} (Height × Width × Channels)")
        print(f"Data type: {self.image_array.dtype}")
        print(f"Value range: {self.image_array.min()} - {self.image_array.max()}")
        print(f"Memory usage: {self.image_array.nbytes / 1024:.1f} KB")

        return self.image_array


    def convert_to_grayscale(self, image_array=None):
        """
        Convert color image to grayscale using luminance formula.

        This method demonstrates how mathematical operations on arrays
        transform images. The luminance formula accounts for human perception:
        Y = 0.2989 × R + 0.5870 × G + 0.1140 × B

        Args:
            image_array (np.ndarray, optional): Input image array.
                                              If None, uses the loaded image.

        Returns:
            np.ndarray: Grayscale image as 2D array (Height, Width)

        Raises:
            ValueError: If no image is loaded and image_array is None
        """
        # Use provided image or loaded image
        if image_array is None:
            image_array = self.image_array

        if image_array is None:
            raise ValueError("No image loaded. Call load_image() first or provide image_array.")

        # Validate input is a color image (3D array)
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(f"Expected 3-channel color image, got shape {image_array.shape}")

        print(f"Converting image to grayscale...")
        print(f"Input shape: {image_array.shape}")

        # Extract individual color channels using array slicing
        # This demonstrates how we can access specific parts of the 3D array
        red_channel = image_array[:, :, 0]    # All rows, all columns, red channel (0)
        green_channel = image_array[:, :, 1]  # All rows, all columns, green channel (1)
        blue_channel = image_array[:, :, 2]   # All rows, all columns, blue channel (2)

        # Apply the luminance formula using vectorized operations (BROADCASTING!)
        # WHY: This formula accounts for human perception of brightness.
        # Different colors contribute differently to perceived brightness.
        # This is a perfect example of how vectorization makes array operations efficient.
        grayscale = (0.2989 * red_channel +
                     0.5870 * green_channel +
                     0.1140 * blue_channel)

        # Ensure values stay in valid range [0, 255] and convert to proper dtype
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)

        print(f"Output shape: {grayscale.shape} (2D - Height × Width)")
        print("Grayscale conversion completed!")

        return grayscale


    def invert_colors(self, image_array=None):
        """
        Invert colors using: New_Pixel = 255 - Old_Pixel.

        This demonstrates element-wise arithmetic operations on the entire array
        using vectorization (BROADCASTING!).

        Args:
            image_array (np.ndarray, optional): Input image array.
                                              If None, uses the loaded image.

        Returns:
            np.ndarray: Color-inverted image array with same shape as input

        Raises:
            ValueError: If no image is loaded and image_array is None
        """
        # Use provided image or loaded image
        if image_array is None:
            image_array = self.image_array

        if image_array is None:
            raise ValueError("No image loaded. Call load_image() first or provide image_array.")

        print(f"Inverting image colors...")
        print(f"Input shape: {image_array.shape}")

        # Apply color inversion using vectorized operation (BROADCASTING!)
        # WHY: This works on the entire array without loops, leveraging C optimizations
        # The scalar value 255 is broadcast to match the image array dimensions
        inverted = 255 - image_array

        print(f"Output shape: {inverted.shape}")
        print("Color inversion completed!")

        return inverted

    def display_comparison(self, original, filtered, title="Comparison"):
        """
        Display original and filtered images side by side.

        Args:
            original: Original image array
            filtered: Processed image array
            title (str): Title for the comparison
        """
        if not MATPLOTLIB_AVAILABLE:
            print(f"Display not available: matplotlib not installed")
            print(f"Comparison: {title}")
            print(f"Original shape: {original.shape}, Filtered shape: {filtered.shape}")
            return

        # Handle both color and grayscale images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Display original image
        if len(original.shape) == 3:  # Color image
            ax1.imshow(original)
        else:  # Grayscale image
            ax1.imshow(original, cmap='gray')
        ax1.set_title("Original")
        ax1.axis('off')  # Hide axes for cleaner look

        # Display filtered image
        if len(filtered.shape) == 3:  # Color image
            ax2.imshow(filtered)
        else:  # Grayscale image
            ax2.imshow(filtered, cmap='gray')
        ax2.set_title("Filtered")
        ax2.axis('off')  # Hide axes for cleaner look

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def save_image(self, image_array, output_path):
        """
        Save NumPy array as image file.

        Args:
            image_array (np.ndarray): Image to save
            output_path (str): Output file path
        """
        # Ensure proper data type and range for image saving
        if image_array.dtype != np.uint8:
            # Clip values to valid range [0, 255] and convert to uint8
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image and save
        if len(image_array.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image_array, mode='L')
        else:  # Color image
            pil_image = Image.fromarray(image_array, mode='RGB')

        pil_image.save(output_path)
        print(f"Image saved to: {output_path}")


def demonstrate_mini_project():
    """
    Main demonstration function for the image processing mini-project.

    This function showcases all implemented functionality and demonstrates
    how images are processed as NumPy arrays.
    """
    print("=" * 70)
    print("IMAGE PROCESSING MINI-PROJECT - DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demonstrates that images are 3D NumPy arrays (H × W × C)")
    print("and filters are mathematical operations on these arrays.")
    print()

    # Initialize processor
    processor = ImageFilters()

    # Use test image (in practice, load with: processor.load_image("path/to/image.jpg"))
    print("Creating test image for demonstration...")
    test_image = create_test_image()
    processor.image_array = test_image
    processor.original_shape = test_image.shape

    print(f"Test image created with shape: {test_image.shape}")
    print()

    # Grayscale conversion
    print("1. Grayscale Conversion")
    print("-" * 30)
    grayscale = processor.convert_to_grayscale()
    print(f"Grayscale shape: {grayscale.shape} (now 2D - Height × Width)")
    print()

    # Color inversion
    print("2. Color Inversion")
    print("-" * 30)
    inverted = processor.invert_colors()
    print(f"Inverted shape: {inverted.shape} (still 3D - H × W × C)")
    print()

    # Display results (if matplotlib is available)
    print("3. Displaying Results")
    print("-" * 30)
    try:
        processor.display_comparison(test_image, grayscale, "Original vs Grayscale")
        processor.display_comparison(test_image, inverted, "Original vs Inverted")
        print("Visual comparisons displayed successfully!")
    except Exception as e:
        print(f"Could not display images (matplotlib issue): {e}")
        print("But calculations were successful!")

    print()
    print("✅ Mini-Project Demonstration Complete!")
    print()
    print("KEY TAKEAWAY:")
    print("- All operations were mathematical transformations")
    print("  on the NumPy array representation of the image")
    print("- Vectorization and broadcasting made operations efficient")
    print("- Images are fundamentally arrays of numbers")


if __name__ == "__main__":
    print("=" * 70)
    print("IMAGE PROCESSING MINI-PROJECT - NUMPY MASTERY")
    print("=" * 70)
    print()
    print("This project demonstrates that images are 3D NumPy arrays.")
    print("Shape: (Height × Width × Channels)")
    print("Filters are mathematical operations on these arrays.")
    print()

    try:
        demonstrate_mini_project()
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()