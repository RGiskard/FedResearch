"""
q_giskard.utils

Helper functions: serialization, deserialization, metrics.

GISKARD:
U (utility): package utilities
"""
def serialize(params):
    """
    Convert numpy params array to list for serialization.
    """
    return params.tolist()

def deserialize(param_list):
    """
    Convert list back to numpy array.
    """
    import numpy as np
    return np.array(param_list)
