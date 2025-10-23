"""
Summarizes all experiments and their results in the federated learning system.
This utility is for documentation and overview purposes only.
"""

EXPERIMENTS_OVERVIEW = [
    {
        "name": "Label Flipping Attack Study",
        "description": "Analyzes the impact of label flipping attacks on federated learning models.",
        "results": "See results/ for CSV and JSON summaries."
    },
    {
        "name": "Timing Attack Analysis",
        "description": "Studies how timing of attacks affects model convergence and accuracy.",
        "results": "Refer to attack_timing.py outputs."
    },
    {
        "name": "Malicious Participant Availability",
        "description": "Explores the effect of malicious clients on overall system robustness.",
        "results": "Refer to malicious_participant_availability.py outputs."
    }
]

def get_experiments_overview():
    """Returns a list of experiment summaries."""
    return EXPERIMENTS_OVERVIEW

if __name__ == "__main__":
    for exp in get_experiments_overview():
        print(f"Experiment: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"Results: {exp['results']}\n")
