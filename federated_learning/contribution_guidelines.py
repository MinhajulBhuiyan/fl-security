"""
Guidelines for contributing new research modules to the federated learning security project.
"""

CONTRIBUTION_GUIDELINES = [
    "Write clear docstrings for all new modules and functions.",
    "Ensure new code does not break existing build or deployment.",
    "Add tests or example usage for new features if possible.",
    "Document any new experiment scripts in experiments_overview.py.",
    "Follow PEP8 style for Python code.",
    "Use descriptive commit messages for all changes.",
    "Submit pull requests with a summary of changes and motivation."
]

def get_contribution_guidelines():
    """Returns guidelines for contributing to the project."""
    return CONTRIBUTION_GUIDELINES

if __name__ == "__main__":
    for guideline in get_contribution_guidelines():
        print(f"- {guideline}")
