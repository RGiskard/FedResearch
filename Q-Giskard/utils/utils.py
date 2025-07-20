"""
q_giskard.utils

Utility helpers: serialization, deserialization, logging, metrics.
"""
def serialize(params):
    # Convert numpy array to list for JSON serialization
    return params.tolist()

def deserialize(param_list):
    # Convert list back to numpy array
    import numpy as np
    return np.array(param_list)
