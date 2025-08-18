# -*- coding: utf-8 -*-

"""
========================================================================
      VISUAL DEMONSTRATION OF THE NO-CLONING THEOREM (FILE SAVING VERSION)
========================================================================

### INTRODUCTION: WHAT IS THE NO-CLONING THEOREM? ###

The No-Cloning Theorem is a fundamental principle in quantum mechanics stating
that it is IMPOSSIBLE to create an identical, independent copy of an
arbitrary, unknown quantum state (qubit).

This script provides a visual demonstration by:
1. Visualizing single qubit states on the 3D Bloch Sphere.
2. Performing a numerical "proof by contradiction".
3. Visualizing the results, showing graphically why a universal cloner fails.

This version is designed for non-interactive environments (like WSL/terminals)
and saves the plots as image files instead of trying to display them.

### KEY REFERENCES ###
- Wootters, W. K., & Zurek, W. H. (1982). A single quantum cannot be cloned. Nature.
- Dieks, D. (1982). Communication by EPR devices. Physics Letters A.

"""

# --- 1. SETUP ---
import numpy as np
import matplotlib
# We explicitly set a non-interactive backend to prevent errors in terminal-only environments.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 2. VISUALIZATION HELPER FUNCTIONS ---

def plot_bloch_sphere(ax, state_vector, title):
    """Plots a qubit state vector on a 3D Bloch Sphere."""
    # Draw the sphere wireframe
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.15)

    # Draw axes and labels
    ax.quiver(0, 0, 0, 0, 0, 1.3, color="k", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, -1.3, color="k", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 1.3, 0, 0, color="gray", arrow_length_ratio=0.1, ls='--')
    ax.quiver(0, 0, 0, -1.3, 0, 0, color="gray", arrow_length_ratio=0.1, ls='--')
    ax.quiver(0, 0, 0, 0, 1.3, 0, color="gray", arrow_length_ratio=0.1, ls='--')
    ax.quiver(0, 0, 0, 0, -1.3, 0, color="gray", arrow_length_ratio=0.1, ls='--')
    ax.text(0, 0, 1.4, r'$|0\rangle$')
    ax.text(0, 0, -1.5, r'$|1\rangle$')
    ax.text(1.4, 0, 0, r'$|+\rangle$')
    ax.text(0, 1.4, 0, r'$|i\rangle$')


    # Calculate Bloch vector coordinates from state vector [alpha, beta]
    alpha = state_vector[0][0]
    beta = state_vector[1][0]
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)
    bx, by, bz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)

    # Draw the state vector
    ax.quiver(0, 0, 0, bx, by, bz, color="r", length=1.0, arrow_length_ratio=0.2, lw=2)
    
    # Formatting
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X-axis"); ax.set_ylabel("Y-axis"); ax.set_zlabel("Z-axis")
    ax.set_aspect('equal')

def plot_state_probabilities(ax, state_vector, title):
    """Creates a bar chart of the measurement probabilities for a quantum state."""
    probabilities = np.abs(state_vector.flatten())**2
    num_qubits = int(np.log2(len(probabilities)))
    labels = [f'$|{i:0{num_qubits}b}\\rangle$' for i in range(len(probabilities))]
    
    ax.bar(labels, probabilities, color='deepskyblue', edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_xlabel("Basis State", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i, p in enumerate(probabilities):
        if p > 0.001:
            ax.text(i, p + 0.03, f'{p:.2f}', ha='center', fontsize=12, weight='bold')


# ========================================================================
# --- SCRIPT EXECUTION STARTS HERE ---
# ========================================================================

# 1. VISUALIZING SINGLE QUBIT STATES
print("--- Generating Plot 1: Visualizing single qubit states ---")
q_zero = np.array([[1], [0]], dtype=complex)
q_plus = (1 / np.sqrt(2)) * (q_zero + np.array([[0], [1]], dtype=complex))
fig_bloch = plt.figure(figsize=(12, 6))
fig_bloch.suptitle("Single Qubit State Visualization", fontsize=16)
ax1 = fig_bloch.add_subplot(121, projection='3d')
ax2 = fig_bloch.add_subplot(122, projection='3d')
plot_bloch_sphere(ax1, q_zero, "State |0⟩")
plot_bloch_sphere(ax2, q_plus, "State |+⟩")
plt.tight_layout(rect=[0, 0.03, 1, 0.93])

# Save the first figure to a file instead of showing it
output_filename_1 = "1_bloch_sphere_visualization.png"
plt.savefig(output_filename_1, dpi=150, bbox_inches='tight')
plt.close(fig_bloch) # Close the figure to free memory
print(f"SUCCESS: Plot 1 saved as '{output_filename_1}'")


# 2. THE NO-CLONING PROOF (NUMERICAL & VISUAL)
print("\n--- Generating Plot 2: The Visual Proof of No-Cloning ---")
# Theory: If a cloner is a linear operator, cloning |+> results in an entangled state.
# This is different from the desired product state.
# ACTUAL result = (1/√2)(|00> + |11>)
# DESIRED result = |+> ⊗ |+>

# Calculate the ACTUAL result
q_00 = np.kron(q_zero, q_zero)
q_11 = np.kron(np.array([[0],[1]]), np.array([[0],[1]]))
actual_result = (1 / np.sqrt(2)) * (q_00 + q_11)

# Calculate the DESIRED result
desired_result = np.kron(q_plus, q_plus)

# Numerical Comparison (printed to console)
print("  - Numerically comparing the outcomes...")
are_equal = np.allclose(actual_result, desired_result)
print(f"  - Are the actual and desired results numerically equal? -> {are_equal}")

# Visual Comparison
fig_proof, (ax_actual, ax_desired) = plt.subplots(1, 2, figsize=(14, 7))
fig_proof.suptitle("Visual Proof of the No-Cloning Theorem", fontsize=18, weight='bold')
plot_state_probabilities(ax_actual, actual_result, "ACTUAL Result (from a Linear Operator)")
plot_state_probabilities(ax_desired, desired_result, "DESIRED Result (from a Perfect Clone)")

# Add explanatory text to the figure to make it more descriptive
explanation_text = "The bar charts show the probabilities of measuring each basis state.\nNotice they are fundamentally different. This contradiction proves that a universal cloner is impossible."
fig_proof.text(0.5, 0.02, explanation_text, ha='center', style='italic', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.4))
plt.tight_layout(rect=[0, 0.1, 1, 0.93])

# Save the second figure to a file
output_filename_2 = "2_no_cloning_proof_visualization.png"
plt.savefig(output_filename_2, dpi=150, bbox_inches='tight')
plt.close(fig_proof) # Close the figure
print(f"SUCCESS: Plot 2 saved as '{output_filename_2}'")

print("\n--- Script finished successfully. Check the generated PNG files. ---")
