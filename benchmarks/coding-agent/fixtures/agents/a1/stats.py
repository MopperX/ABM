def median(values):
    """Return the median without mutating the input list."""
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
