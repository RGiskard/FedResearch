"""
q_giskard.preprocessing.decoding

Quantum → Classical decoding:
Processes measurement outcomes into classical feature vectors.

References:
- Huang, H.-Y., Kueng, R., & Preskill, J. (2020). Predicting many properties of a quantum system from very few measurements. Nature Physics 16, 1050–1057.
"""
def decode_measurements(measurements):
    """
    Convert expectation values or raw measurement outcomes into classical data.
    Args:
        measurements (dict or array): raw results or expectation values
    Returns:
        array: classical feature vector
    """
    # TODO: average shots and map Pauli-Z expectations to feature vector
    pass
