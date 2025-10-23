"""
Example worker selection strategies for federated learning, with pseudocode.
"""

WORKER_SELECTION_EXAMPLES = [
    {
        "strategy": "Random Selection",
        "pseudocode": "Select N workers at random each round."
    },
    {
        "strategy": "Before Breakpoint",
        "pseudocode": "Select workers with index < breakpoint for early rounds."
    },
    {
        "strategy": "After Breakpoint",
        "pseudocode": "Select workers with index >= breakpoint for later rounds."
    },
    {
        "strategy": "Poisoner Probability",
        "pseudocode": "Select poisoned workers with higher probability."
    }
]

def get_worker_selection_examples():
    """Returns example worker selection strategies."""
    return WORKER_SELECTION_EXAMPLES

if __name__ == "__main__":
    for example in get_worker_selection_examples():
        print(f"Strategy: {example['strategy']}")
        print(f"Pseudocode: {example['pseudocode']}\n")
