"""
Example scenarios for data/model poisoning in federated learning.
Useful for understanding attack vectors and designing defenses.
"""

POISONING_SCENARIOS = [
    {
        "scenario": "Label flipping by 10% of clients",
        "impact": "Global model accuracy drops by 15-30%."
    },
    {
        "scenario": "Random noise injection in client updates",
        "impact": "Model convergence slows, but accuracy remains stable if noise is low."
    },
    {
        "scenario": "Targeted backdoor insertion",
        "impact": "Model misclassifies specific inputs with high confidence."
    }
]

def get_poisoning_scenarios():
    """Returns example poisoning scenarios."""
    return POISONING_SCENARIOS

if __name__ == "__main__":
    for scenario in get_poisoning_scenarios():
        print(f"Scenario: {scenario['scenario']}")
        print(f"Impact: {scenario['impact']}\n")
