"""
Helper functions for visualizing federated learning experiment results.
No external dependencies required; uses simple text-based output.
"""

def plot_accuracy_curve(accuracy_list):
    """Prints a simple accuracy curve as text."""
    print("Accuracy Curve:")
    for epoch, acc in enumerate(accuracy_list, 1):
        bar = '#' * int(acc)
        print(f"Epoch {epoch:3d}: {bar} ({acc:.2f}%)")

def plot_loss_curve(loss_list):
    """Prints a simple loss curve as text."""
    print("Loss Curve:")
    for epoch, loss in enumerate(loss_list, 1):
        bar = '-' * int(100 - loss * 20)
        print(f"Epoch {epoch:3d}: {bar} (Loss: {loss:.4f})")

if __name__ == "__main__":
    # Example usage
    plot_accuracy_curve([70, 75, 80, 85, 90])
    plot_loss_curve([2.0, 1.5, 1.0, 0.7, 0.5])
