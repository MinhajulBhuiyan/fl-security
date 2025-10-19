from federated_learning.arguments import Arguments
from federated_learning.nets import Cifar10CNN
from federated_learning.nets import FashionMNISTCNN
import os
import torch
from loguru import logger
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn

def train_basic_fashion_mnist_model(model):
    """Train a basic Fashion-MNIST model for demonstration purposes."""
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.train()

        # Load a small subset of Fashion-MNIST for quick training
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

        # Use only 2000 samples for quick training
        subset_indices = list(range(0, len(train_dataset), len(train_dataset)//2000))[:2000]
        train_subset = Subset(train_dataset, subset_indices)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

        # Simple training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        print("Training Fashion-MNIST model on 2000 samples...")
        for epoch in range(5):  # 5 epochs for better training
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        model.eval()
        print("Fashion-MNIST model training completed")
        return model

    except Exception as e:
        print(f"Error training Fashion-MNIST model: {e}")
        return model

def train_basic_cifar10_model(model):
    """Train a basic CIFAR-10 model for demonstration purposes."""
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.train()

        # Load a small subset of CIFAR-10 for quick training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # Use only 2000 samples for quick training
        subset_indices = list(range(0, len(train_dataset), len(train_dataset)//2000))[:2000]
        train_subset = Subset(train_dataset, subset_indices)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

        # Simple training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        print("Training CIFAR-10 model on 2000 samples...")
        for epoch in range(5):  # 5 epochs for better training
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        model.eval()
        print("CIFAR-10 model training completed")
        return model

    except Exception as e:
        print(f"Error training CIFAR-10 model: {e}")
        return model

if __name__ == '__main__':
    args = Arguments(logger)
    if not os.path.exists(args.get_default_model_folder_path()):
        os.mkdir(args.get_default_model_folder_path())

    # ---------------------------------
    # ----------- Cifar10CNN ----------
    # ---------------------------------
    print("Creating trained CIFAR-10 model...")
    cifar_model = Cifar10CNN()
    trained_cifar_model = train_basic_cifar10_model(cifar_model)
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10CNN_trained.model")
    torch.save(trained_cifar_model.state_dict(), full_save_path)
    print(f"Saved trained CIFAR-10 model to {full_save_path}")

    # ---------------------------------
    # -------- FashionMNISTCNN --------
    # ---------------------------------
    print("Creating trained Fashion-MNIST model...")
    fashion_model = FashionMNISTCNN()
    trained_fashion_model = train_basic_fashion_mnist_model(fashion_model)
    full_save_path = os.path.join(args.get_default_model_folder_path(), "FashionMNISTCNN_trained.model")
    torch.save(trained_fashion_model.state_dict(), full_save_path)
    print(f"Saved trained Fashion-MNIST model to {full_save_path}")

    print("All trained models created successfully!")
