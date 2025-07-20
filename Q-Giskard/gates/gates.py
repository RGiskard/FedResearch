"""
q_giskard.gates

Gate-based Quantum Circuits module.
Creates and manages variational circuits.

References:
- Schuld, M., & Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. *Physical Review Letters*, 122(4), 040504.
- Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.

GISKARD:
G - Gate-based Quantum Circuits
"""
def create_variational_circuit(params, n_qubits):
    """
    Build a parameterized quantum circuit.
    Args:
        params (array): circuit parameters [layers, n_qubits]
        n_qubits (int): number of qubits
    Returns:
        QuantumCircuit or QNode
    """
    # TODO: implement layering of Ry, Rz rotations and CNOT entangling gates
    pass
