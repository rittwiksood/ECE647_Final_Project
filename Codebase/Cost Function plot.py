import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generate sample data for visualization
def generate_sample_data():
    np.random.seed(42)
    Nt, Nr, P = 2, 1, 10  # Simplified 2D case for visualization
    Phi = np.random.randn(P, Nt) + 1j * np.random.randn(P, Nt)
    Phi /= np.linalg.norm(Phi, axis=0)
    H_true = np.array([[1 + 0.5j], [0 + 0j]])  # One non-zero entry
    y = Phi @ H_true
    return Phi, y, H_true


# LASSO cost function
def lasso_cost(h, y, Phi, lambda_lasso):
    return np.linalg.norm(y - Phi @ h) ** 2 + lambda_lasso * np.linalg.norm(h, 1)


# Group LASSO cost function (2D case with single group)
def group_lasso_cost(h, y, Phi, lambda_glasso):
    return np.linalg.norm(y - Phi @ h) ** 2 + lambda_glasso * np.linalg.norm(h)


# OMP cost function (non-convex)
def omp_cost(h, y, Phi):
    return np.linalg.norm(y - Phi @ h) ** 2


# Create grid for visualization
def create_visualization_grid():
    real_vals = np.linspace(-2, 2, 50)
    imag_vals = np.linspace(-2, 2, 50)
    R, I = np.meshgrid(real_vals, imag_vals)
    H_grid = np.zeros((50, 50, 2, 1), dtype=complex)
    H_grid[:, :, 0, 0] = R + 1j * I
    H_grid[:, :, 1, 0] = R.T + 1j * I.T
    return R, I, H_grid


# Plot 3D surface of cost functions
def plot_cost_functions():
    Phi, y, H_true = generate_sample_data()
    R, I, H_grid = create_visualization_grid()
    lambda_lasso = 0.5
    lambda_glasso = 0.5

    # Prepare cost function values
    lasso_values = np.zeros_like(R, dtype=float)
    group_lasso_values = np.zeros_like(R, dtype=float)
    omp_values = np.zeros_like(R, dtype=float)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            h = H_grid[i, j]
            lasso_values[i, j] = lasso_cost(h, y, Phi, lambda_lasso)
            group_lasso_values[i, j] = group_lasso_cost(h, y, Phi, lambda_glasso)
            omp_values[i, j] = omp_cost(h, y, Phi)

    # Create figure
    fig = plt.figure(figsize=(18, 6))

    # LASSO plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(R, I, lasso_values, cmap='viridis', alpha=0.8)
    ax1.set_title('LASSO Cost Function (Convex)', fontsize=12)
    ax1.set_xlabel('Re(h₁)')
    ax1.set_ylabel('Im(h₁)')
    ax1.set_zlabel('Cost')
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Group LASSO plot
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(R, I, group_lasso_values, cmap='plasma', alpha=0.8)
    ax2.set_title('Group LASSO Cost Function (Convex)', fontsize=12)
    ax2.set_xlabel('Re(h₁)')
    ax2.set_ylabel('Im(h₁)')
    ax2.set_zlabel('Cost')
    ax2.view_init(elev=30, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # OMP plot
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(R, I, omp_values, cmap='magma', alpha=0.8)
    ax3.set_title('OMP Cost Function (Non-Convex)', fontsize=12)
    ax3.set_xlabel('Re(h₁)')
    ax3.set_ylabel('Im(h₁)')
    ax3.set_zlabel('Cost')
    ax3.view_init(elev=30, azim=45)
    fig.colorbar(surf3, ax=ax3, shrink=0.5)

    plt.tight_layout()
    plt.show()


# Generate and show the plots
plot_cost_functions()

