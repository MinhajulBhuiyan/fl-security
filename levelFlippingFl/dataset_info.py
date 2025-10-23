"""
Information and docstrings about supported datasets in federated learning experiments.
"""

DATASET_INFO = {
    "fashion_mnist": {
        "description": "Fashion-MNIST: 28x28 grayscale images of clothing items, 10 classes.",
        "source": "https://github.com/zalandoresearch/fashion-mnist"
    },
    "cifar10": {
        "description": "CIFAR-10: 32x32 color images, 10 classes (objects and animals).",
        "source": "https://www.cs.toronto.edu/~kriz/cifar.html"
    }
}

def get_dataset_info(name):
    """Returns info for a given dataset."""
    return DATASET_INFO.get(name, "Dataset not found.")

if __name__ == "__main__":
    for dataset, info in DATASET_INFO.items():
        print(f"Dataset: {dataset}")
        print(f"Description: {info['description']}")
        print(f"Source: {info['source']}\n")
