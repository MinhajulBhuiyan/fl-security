def generate_experiment_ids(start_idx, num_exp):
    """
    Generate the filenames for all experiment IDs.

    :param start_idx: start index for experiments (can be int or string experiment ID)
    :type start_idx: int or str
    :param num_exp: number of experiments to run
    :type num_exp: int
    """
    log_files = []
    results_files = []
    models_folders = []
    worker_selections_files = []

    for i in range(num_exp):
        # Handle both string experiment IDs and integer indices
        if isinstance(start_idx, str):
            # If start_idx is a string (experiment ID), use it directly for the first experiment
            if i == 0:
                idx = start_idx
            else:
                # For multiple experiments, append suffix
                idx = f"{start_idx}_{i}"
        else:
            # If start_idx is an integer, use the original logic
            idx = str(start_idx + i)

        log_files.append("logs/" + idx + ".log")
        results_files.append("results/" + idx + "_results.csv")
        models_folders.append(idx + "_models")
        worker_selections_files.append("results/" + idx + "_workers_selected.csv")

    return log_files, results_files, models_folders, worker_selections_files
