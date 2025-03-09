def optimal_inspection_time(defects, max_allowed_defects):
    """
    Finds the optimal stopping time for defect inspections.
    
    Args:
        defects: Cumulative defect count over time.
        max_allowed_defects: Threshold to stop inspections.

    Returns:
        Optimal inspection stopping time.
    """
    for t, count in enumerate(defects):
        if count >= max_allowed_defects:
            return t
    return len(defects)  # Inspect all batches
