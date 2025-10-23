"""
Security best practices for federated learning systems.
This file provides tips to improve robustness against attacks.
"""

SECURITY_TIPS = [
    "Validate client updates before aggregation to detect anomalies.",
    "Use differential privacy to protect client data.",
    "Limit the influence of any single client on the global model.",
    "Monitor training metrics for sudden drops in accuracy or spikes in loss.",
    "Implement robust worker selection strategies to avoid repeated selection of malicious clients.",
    "Encrypt communication between server and clients.",
    "Regularly retrain models with fresh data to reduce stale attack effects."
]

def get_security_tips():
    """Returns a list of security best practices for FL."""
    return SECURITY_TIPS

if __name__ == "__main__":
    for tip in get_security_tips():
        print(f"- {tip}")
