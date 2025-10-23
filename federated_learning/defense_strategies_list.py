"""
List and brief explanation of defense strategies against attacks in federated learning.
"""

DEFENSE_STRATEGIES = [
    {
        "name": "Robust Aggregation",
        "description": "Use aggregation methods (e.g., median, trimmed mean) to reduce impact of outliers."
    },
    {
        "name": "Anomaly Detection",
        "description": "Detect and exclude suspicious client updates based on statistical analysis."
    },
    {
        "name": "Differential Privacy",
        "description": "Add noise to client updates to protect privacy and reduce attack effectiveness."
    },
    {
        "name": "Client Reputation Systems",
        "description": "Track client behavior and reduce influence of consistently malicious clients."
    }
]

def get_defense_strategies():
    """Returns a list of defense strategies."""
    return DEFENSE_STRATEGIES

if __name__ == "__main__":
    for strategy in get_defense_strategies():
        print(f"Strategy: {strategy['name']}")
        print(f"Description: {strategy['description']}\n")
