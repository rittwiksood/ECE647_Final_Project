import numpy as np
import cvxpy as cp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def generate_sparse_channel(Nt, Nr, S, snr_db):
    """
    Generate a sparse MIMO channel matrix
    Args:
        Nt: Number of transmit antennas
        Nr: Number of receive antennas
        S: Number of non-zero paths (sparsity level)
        snr_db: SNR in dB
    Returns:
        H: Sparse channel matrix (Nt x Nr)
        noise: Properly sized noise matrix (P x Nr)
    """
    # Create sparse matrix with S non-zero entries
    H = np.zeros((Nt, Nr), dtype=complex)
    nonzero_indices = np.random.choice(Nt * Nr, S, replace=False)
    values = (np.random.randn(S) + 1j * np.random.randn(S)) / np.sqrt(2)
    H.flat[nonzero_indices] = values

    # Normalize channel power
    H = H / np.linalg.norm(H, 'fro') * np.sqrt(Nt * Nr)

    return H


def lasso_channel_estimation(y, Phi, lambda_lasso):
    """
    LASSO-based channel estimation
    Args:
        y: Received signal (P x Nr)
        Phi: Pilot matrix (P x Nt)
        lambda_lasso: Regularization parameter
    Returns:
        H_est: Estimated channel (Nt x Nr)
    """
    P, Nr = y.shape
    Nt = Phi.shape[1]

    H_est = np.zeros((Nt, Nr), dtype=complex)

    for i in range(Nr):
        # Solve LASSO for each receive antenna
        h = cp.Variable(Nt, complex=True)
        objective = cp.Minimize(cp.sum_squares(y[:, i] - Phi @ h) + lambda_lasso * cp.norm1(h))
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS)

        H_est[:, i] = h.value

    return H_est


def group_lasso_channel_estimation(y, Phi, lambda_glasso, group_size):
    """
    Group LASSO-based channel estimation
    Args:
        y: Received signal (P x Nr)
        Phi: Pilot matrix (P x Nt)
        lambda_glasso: Regularization parameter
        group_size: Size of antenna correlation groups
    Returns:
        H_est: Estimated channel (Nt x Nr)
    """
    P, Nr = y.shape
    Nt = Phi.shape[1]
    num_groups = Nt // group_size

    H_est = np.zeros((Nt, Nr), dtype=complex)

    for i in range(Nr):
        # Solve Group LASSO for each receive antenna
        h = cp.Variable(Nt, complex=True)

        # Create group norms
        group_norms = []
        for g in range(num_groups):
            group = h[g * group_size:(g + 1) * group_size]
            group_norms.append(cp.norm2(group))

        objective = cp.Minimize(cp.sum_squares(y[:, i] - Phi @ h) +
                                lambda_glasso * cp.sum(group_norms))
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS)

        H_est[:, i] = h.value

    return H_est


def omp_channel_estimation(y, Phi, sparsity):
    """
    Orthogonal Matching Pursuit for channel estimation
    Args:
        y: Received signal (P x Nr)
        Phi: Pilot matrix (P x Nt)
        sparsity: Expected sparsity level
    Returns:
        H_est: Estimated channel (Nt x Nr)
    """
    P, Nr = y.shape
    Nt = Phi.shape[1]

    H_est = np.zeros((Nt, Nr), dtype=complex)

    for i in range(Nr):
        residual = y[:, i].copy()
        idx_set = []
        Phi_r = Phi.copy()

        for _ in range(sparsity):
            # Find the column of Phi most correlated with residual
            correlations = np.abs(Phi_r.T.conj() @ residual)
            new_idx = np.argmax(correlations)
            idx_set.append(new_idx)

            # Solve least squares on selected columns
            Phi_active = Phi[:, idx_set]
            h_ls = np.linalg.pinv(Phi_active) @ y[:, i]

            # Update residual
            residual = y[:, i] - Phi_active @ h_ls

            # Remove selected column from consideration
            Phi_r[:, new_idx] = 0

        # Reconstruct the sparse vector
        h_est = np.zeros(Nt, dtype=complex)
        h_est[idx_set] = h_ls
        H_est[:, i] = h_est

    return H_est


def evaluate_performance(H_true, H_est):
    """
    Evaluate estimation performance
    Args:
        H_true: True channel matrix
        H_est: Estimated channel matrix
    Returns:
        nmse: Normalized MSE
        support_error: Support recovery error
    """
    # Normalized MSE
    nmse = np.linalg.norm(H_est - H_true, 'fro') ** 2 / np.linalg.norm(H_true, 'fro') ** 2

    # Support recovery error
    true_support = np.abs(H_true) > 1e-3
    est_support = np.abs(H_est) > 1e-3
    support_error = np.mean(true_support != est_support)

    return nmse, support_error


# Simulation parameters
Nt = 64  # Number of transmit antennas
Nr = 8  # Number of receive antennas
S = 8  # Sparsity level
P = 16  # Number of pilots
snr_db = 20  # SNR in dB
lambda_lasso = 0.1  # LASSO regularization
lambda_glasso = 0.1  # Group LASSO regularization
group_size = 4  # Antenna group size
num_trials = 100  # Number of Monte Carlo trials

# Storage for results
nmse_lasso = np.zeros(num_trials)
nmse_glasso = np.zeros(num_trials)
nmse_omp = np.zeros(num_trials)
support_lasso = np.zeros(num_trials)
support_glasso = np.zeros(num_trials)
support_omp = np.zeros(num_trials)

# Generate random pilot matrix
Phi = (np.random.randn(P, Nt) + 1j * np.random.randn(P, Nt)) / np.sqrt(2 * P)

for trial in range(num_trials):
    # Generate sparse channel
    H_true = generate_sparse_channel(Nt, Nr, S, snr_db)

    # Generate properly sized noise matrix (P x Nr)
    sigma2 = 10 ** (-snr_db / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(P, Nr) + 1j * np.random.randn(P, Nr))

    y = Phi @ H_true + noise

    # LASSO estimation
    H_lasso = lasso_channel_estimation(y, Phi, lambda_lasso)
    nmse_lasso[trial], support_lasso[trial] = evaluate_performance(H_true, H_lasso)

    # Group LASSO estimation
    H_glasso = group_lasso_channel_estimation(y, Phi, lambda_glasso, group_size)
    nmse_glasso[trial], support_glasso[trial] = evaluate_performance(H_true, H_glasso)

    # OMP estimation
    H_omp = omp_channel_estimation(y, Phi, S)
    nmse_omp[trial], support_omp[trial] = evaluate_performance(H_true, H_omp)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.boxplot([nmse_lasso, nmse_glasso, nmse_omp], tick_labels=['LASSO', 'Group LASSO', 'OMP'])
plt.yscale('log')
plt.ylabel('NMSE')
plt.title('Normalized Mean Squared Error')

plt.subplot(1, 2, 2)
plt.boxplot([support_lasso, support_glasso, support_omp], tick_labels=['LASSO', 'Group LASSO', 'OMP'])
plt.ylabel('Support Recovery Error')
plt.title('Support Recovery Performance')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.yscale('log')
plt.boxplot([nmse_lasso], tick_labels=['LASSO'])
plt.ylabel('NMSE')
plt.title('Normalized Mean Squared Error')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.yscale('log')
plt.boxplot([nmse_glasso], tick_labels=['Group LASSO'])
plt.ylabel('NMSE')
plt.title('Normalized Mean Squared Error')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.yscale('log')
plt.boxplot([nmse_omp], tick_labels=['OMP'])
plt.ylabel('NMSE')
plt.title('Normalized Mean Squared Error')
plt.tight_layout()
plt.show()

# Print average performance
print(f"Average NMSE - LASSO: {np.mean(nmse_lasso):.4f}")
print(f"Average NMSE - Group LASSO: {np.mean(nmse_glasso):.4f}")
print(f"Average NMSE - OMP: {np.mean(nmse_omp):.4f}")
print(f"Average Support Error - LASSO: {np.mean(support_lasso):.4f}")
print(f"Average Support Error - Group LASSO: {np.mean(support_glasso):.4f}")
print(f"Average Support Error - OMP: {np.mean(support_omp):.4f}")