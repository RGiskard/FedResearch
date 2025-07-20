"""
q_giskard.decoding

Quantum-to-classical decoding:
- Measurement processing
- Expectation value extraction

References:
- Huang, H. Y., Kueng, R., & Preskill, J. (2020). Predicting many properties of a quantum system from very few measurements. *Nature Physics*, 16(10), 1050–1057.

GISKARD:
D - Distribution & Deployment
"""
def decode_measurements(measurements):
    """
    Convert raw measurement results into classical feature vector.
    Args:
        measurements (dict or array): measurement outcomes or expectation values
    Returns:
        array: classical vector
    """
    # TODO: average shots, map Pauli-Z expectations to features
    pass
