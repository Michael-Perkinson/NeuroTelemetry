import math

def calculate_dynamic_bins(array_length: int, b_ref: int = 600, t_ref: int = 1200, k: int = 5) -> int:
    """Dynamically calculate the number of bins for the histogram using logarithmic scaling."""

    bins = int(b_ref * math.log(1 + (array_length * k / t_ref)))
    return min(array_length, bins)
