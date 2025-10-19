"""
Download sample test images for testing the image classification feature.
This script downloads sample images from both Fashion-MNIST and CIFAR-10 datasets.
"""

import os
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image

def download_fashion_mnist_samples(output_dir="test_images/fashion_mnist", num_samples=70):
    """Download sample Fashion-MNIST images"""
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading Fashion-MNIST samples...")
    dataset = datasets.FashionMNIST(root='./data', train=False, download=True)

    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]

    # Get multiple samples from each class
    class_counts = {label: 0 for label in range(10)}
    idx = 0

    while sum(class_counts.values()) < num_samples and idx < len(dataset):
        image, label = dataset[idx]

        if class_counts[label] < 7:  # Save up to 7 images per class
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = transforms.ToPILImage()(image)

            # Save image with consistent naming: ClassName_label_index.png
            filename = f"{class_names[label]}_{label}_{class_counts[label] + 1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"  Saved: {filename}")
            class_counts[label] += 1

        idx += 1

    print(f"✓ Saved {sum(class_counts.values())} Fashion-MNIST test images to {output_dir}/")


def download_cifar10_samples(output_dir="test_images/cifar10", num_samples=70):
    """Download sample CIFAR-10 images"""
    os.makedirs(output_dir, exist_ok=True)

    print("\nDownloading CIFAR-10 samples...")
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)

    class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]

    # Get multiple samples from each class
    class_counts = {label: 0 for label in range(10)}
    idx = 0

    while sum(class_counts.values()) < num_samples and idx < len(dataset):
        image, label = dataset[idx]

        if class_counts[label] < 7:  # Save up to 7 images per class
            # Save image with consistent naming: ClassName_label_index.png
            filename = f"{class_names[label]}_{label}_{class_counts[label] + 1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"  Saved: {filename}")
            class_counts[label] += 1

        idx += 1

    print(f"✓ Saved {sum(class_counts.values())} CIFAR-10 test images to {output_dir}/")


def create_info_file():
    """This function has been removed as the README.md file is no longer needed."""
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("  Downloading Test Images for FL Security Research")
    print("=" * 60)
    print()
    
    # Download Fashion-MNIST samples
    download_fashion_mnist_samples()
    
    # Download CIFAR-10 samples
    download_cifar10_samples()
    
    # Create info file
    create_info_file()
    
    print("\n" + "=" * 60)
    print("  ✓ All test images downloaded successfully!")
    print("=" * 60)
    print()
    print("Location: test_images/")
    print("  - fashion_mnist/ (70 images - 7 per class)")
    print("  - cifar10/ (70 images - 7 per class)")
    print()
    print("You can now use these images to test your trained models!")
    print("=" * 60)
