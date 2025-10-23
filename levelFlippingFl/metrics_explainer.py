"""
Explains key metrics used in federated learning experiments: accuracy, loss, precision, recall.
Useful for understanding experiment results and model evaluation.
"""

METRICS_INFO = {
    "accuracy": "Proportion of correct predictions out of total predictions. Indicates overall model performance.",
    "loss": "A measure of model error during training. Lower loss generally means better fit.",
    "precision": "Fraction of relevant instances among retrieved instances. Important for imbalanced datasets.",
    "recall": "Fraction of relevant instances that were retrieved. Indicates model's ability to find all positives."
}

def explain_metric(metric_name):
    """Returns explanation for a given metric."""
    return METRICS_INFO.get(metric_name, "Metric not found.")

if __name__ == "__main__":
    for metric, info in METRICS_INFO.items():
        print(f"{metric.title()}: {info}\n")
