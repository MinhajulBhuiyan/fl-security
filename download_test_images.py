"""
Download sample test images for testing the image classification feature.
This script downloads sample images from both Fashion-MNIST and CIFAR-10 datasets.
"""

import os
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image

def download_fashion_mnist_samples(output_dir="test_images/fashion_mnist", num_samples=10):
    """Download sample Fashion-MNIST images"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading Fashion-MNIST samples...")
    dataset = datasets.FashionMNIST(root='./data', train=False, download=True)
    
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]
    
    # Get one sample from each class
    saved_classes = set()
    idx = 0
    
    while len(saved_classes) < 10 and idx < len(dataset):
        image, label = dataset[idx]
        
        if label not in saved_classes:
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = transforms.ToPILImage()(image)
            
            # Save image
            filename = f"{class_names[label]}_{label}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"  Saved: {filename}")
            saved_classes.add(label)
        
        idx += 1
    
    print(f"✓ Saved {len(saved_classes)} Fashion-MNIST test images to {output_dir}/")


def download_cifar10_samples(output_dir="test_images/cifar10", num_samples=10):
    """Download sample CIFAR-10 images"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nDownloading CIFAR-10 samples...")
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    
    class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]
    
    # Get one sample from each class
    saved_classes = set()
    idx = 0
    
    while len(saved_classes) < 10 and idx < len(dataset):
        image, label = dataset[idx]
        
        if label not in saved_classes:
            # Save image
            filename = f"{class_names[label]}_{label}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"  Saved: {filename}")
            saved_classes.add(label)
        
        idx += 1
    
    print(f"✓ Saved {len(saved_classes)} CIFAR-10 test images to {output_dir}/")


def create_info_file():
    """Create an info file explaining the test images"""
    info_content = """# Test Images Guide

## Fashion-MNIST Images (28x28 grayscale)
Located in: test_images/fashion_mnist/

Classes:
- T-shirt_0.png - T-shirt/top
- Trouser_1.png - Trouser
- Pullover_2.png - Pullover
- Dress_3.png - Dress
- Coat_4.png - Coat
- Sandal_5.png - Sandal
- Shirt_6.png - Shirt
- Sneaker_7.png - Sneaker
- Bag_8.png - Bag
- Ankle_boot_9.png - Ankle boot

## CIFAR-10 Images (32x32 RGB)
Located in: test_images/cifar10/

Classes:
- airplane_0.png - Airplane
- automobile_1.png - Automobile
- bird_2.png - Bird
- cat_3.png - Cat
- deer_4.png - Deer
- dog_5.png - Dog
- frog_6.png - Frog
- horse_7.png - Horse
- ship_8.png - Ship
- truck_9.png - Truck

## How to Use:
1. Run a federated learning experiment (with or without attack)
2. Go to the "Image Testing" tab in the results
3. Upload any of these test images
4. See how the model classifies it!

## Testing Attack Impact:
1. Run experiment with 0 poisoned workers → Upload bird_2.png → Should classify as "bird" ✓
2. Run experiment with 20 poisoned workers → Upload bird_2.png → Will misclassify ✗
3. Compare the results to see the attack's impact!
"""
    
    with open("test_images/README.md", "w") as f:
        f.write(info_content)
    
    print("\n✓ Created test_images/README.md")


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
    print("  - fashion_mnist/ (10 images)")
    print("  - cifar10/ (10 images)")
    print("  - README.md (usage guide)")
    print()
    print("You can now use these images to test your trained models!")
    print("=" * 60)
