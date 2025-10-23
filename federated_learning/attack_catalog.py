"""
Catalog of attack types used in federated learning security research.
Provides descriptions and references for each attack.
"""

ATTACK_CATALOG = [
    {
        "name": "Label Flipping Attack",
        "description": "Malicious clients flip labels in their local dataset to degrade global model accuracy.",
        "reference": "https://arxiv.org/abs/1803.07453"
    },
    {
        "name": "Data Poisoning Attack",
        "description": "Attackers inject corrupted data to mislead model training.",
        "reference": "https://arxiv.org/abs/1708.08689"
    },
    {
        "name": "Model Poisoning Attack",
        "description": "Malicious updates are sent to the server to manipulate the global model.",
        "reference": "https://arxiv.org/abs/1804.07890"
    },
    {
        "name": "Backdoor Attack",
        "description": "Attackers train models to respond incorrectly to specific triggers.",
        "reference": "https://arxiv.org/abs/1712.05526"
    }
]

def get_attack_catalog():
    """Returns a list of attack types with descriptions."""
    return ATTACK_CATALOG

if __name__ == "__main__":
    for attack in get_attack_catalog():
        print(f"Attack: {attack['name']}")
        print(f"Description: {attack['description']}")
        print(f"Reference: {attack['reference']}\n")
