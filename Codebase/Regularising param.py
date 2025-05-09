import numpy as np
import matplotlib.pyplot as plt


# Generate sample data (real-valued for visualization)
def generate_sample_data():
    np.random.seed(42)
    Nt = 2  # Two transmit antennas for 2D visualization
    P = 10  # Number of pilots
    Phi = np.random.randn(P, Nt)
    Phi /= np.linalg.norm(Phi, axis=0)
    H_true = np.array([[1.0], [0.0]])  # One non-zero entry
    y = Phi @ H_true
    return Phi, y, H_true


# LASSO cost function
def lasso_cost(h, y, Phi, lambda_val):
    return np.linalg.norm(y - Phi @ h.reshape(-1, 1)) ** 2 + lambda_val * np.sum(np.abs(h))


# Create grid for visualization
def create_grid():
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    return X, Y


# Plot LASSO with different regularization parameters in one plot
def plot_lasso_reg_comparison():
    Phi, y, H_true = generate_sample_data()
    X, Y = create_grid()

    # Three different lambda values and colors
    lambda_values = [0.1, 0.5, 1.0]
    colors = ['blue', 'green', 'red']
    line_styles = ['-', '--', ':']
    labels = [f'λ = {lam}' for lam in lambda_values]

    plt.figure(figsize=(10, 8))

    # Plot along h₂=0 line to see 1D cross-section
    h1_values = np.linspace(-2, 2, 200)
    h2_fixed = 0

    for lam, color, ls, label in zip(lambda_values, colors, line_styles, labels):
        cost_values = [lasso_cost(np.array([h1, h2_fixed]), y, Phi, lam) for h1 in h1_values]
        plt.plot(h1_values, cost_values, color=color, linestyle=ls, linewidth=2, label=label)

    # Mark true solution and minima
    plt.scatter([H_true[0]], [0], c='black', marker='*', s=200, label='True Solution')

    for lam, color in zip(lambda_values, colors):
        # Find minimum for this lambda
        min_cost = np.inf
        min_h1 = 0
        for h1 in h1_values:
            current_cost = lasso_cost(np.array([h1, h2_fixed]), y, Phi, lam)
            if current_cost < min_cost:
                min_cost = current_cost
                min_h1 = h1
        plt.scatter([min_h1], [min_cost], color=color, marker='o', s=100)

    plt.title('LASSO Cost Function with Different Regularization Parameters', fontsize=14)
    plt.xlabel('h₁ (with h₂=0)')
    plt.ylabel('Cost Function Value')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)



    plt.tight_layout()
    plt.show()


# Generate and show the plot
plot_lasso_reg_comparison()