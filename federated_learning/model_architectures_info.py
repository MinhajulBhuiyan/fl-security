"""
Information and docstrings about supported model architectures in federated learning experiments.
"""

MODEL_ARCHITECTURES_INFO = {
    "FashionMNISTCNN": {
        "description": "Convolutional neural network for Fashion-MNIST dataset.",
        "input_shape": "(1, 28, 28)",
        "output_classes": 10
    },
    "Cifar10CNN": {
        "description": "Convolutional neural network for CIFAR-10 dataset.",
        "input_shape": "(3, 32, 32)",
        "output_classes": 10
    }
}

def get_model_architecture_info(name):
    """Returns info for a given model architecture."""
    return MODEL_ARCHITECTURES_INFO.get(name, "Model architecture not found.")

if __name__ == "__main__":
    for model, info in MODEL_ARCHITECTURES_INFO.items():
        print(f"Model: {model}")
        print(f"Description: {info['description']}")
        print(f"Input Shape: {info['input_shape']}")
        print(f"Output Classes: {info['output_classes']}\n")
