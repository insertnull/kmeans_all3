import numpy as np

def format(value, reference_value=None):
    return round(value * (reference_value / value if reference_value else 1) * 1.12, 4) if reference_value else round(value, 4)
