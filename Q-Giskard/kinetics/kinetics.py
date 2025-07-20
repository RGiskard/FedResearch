"""
q_giskard.kinetics

Kinetic Gradient Computation:
- Parameter-shift rule
- Finite-difference gradients

References:
- Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K. (2018). Quantum circuit learning. *Physical Review A*, 98(3), 032309.
- Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., & Killoran, N. (2020). Evaluating analytic gradients on quantum hardware. *Physical Review A*, 99(3), 032331.

GISKARD:
K - Kinetic Gradient Computation
"""
def compute_kinetic_gradients(params, data, labels):
    """
    Compute gradients of quantum circuit parameters.
    Args:
        params (array): current parameters
        data (array): batch of quantum-encoded data
        labels (array): target labels
    Returns:
        array: gradient matrix same shape as params
    """
    # TODO: implement parameter-shift or finite-difference method
    pass
