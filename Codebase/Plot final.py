import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import time
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import pandas as pd

# Set the style for the plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (10, 8)


# -----------------------------------------------------------------------------
# Implementation of Algorithms
# -----------------------------------------------------------------------------

def generate_channel(N, K, is_group_sparse=False, group_size=4):
    """
    Generate a sparse channel vector.

    Parameters:
    N: Length of the channel vector
    K: Number of non-zero coefficients (sparsity level)
    is_group_sparse: If True, non-zero elements will be grouped
    group_size: Size of each group if is_group_sparse is True

    Returns:
    h: The sparse channel vector
    support: Indices of non-zero elements
    """
    h = np.zeros(N, dtype=complex)
    if is_group_sparse:
        # Calculate how many groups we need
        num_groups = int(np.ceil(K / group_size))
        # Randomly select group indices
        group_indices = np.random.choice(int(np.ceil(N / group_size)), num_groups, replace=False)
        support = []
        for idx in group_indices:
            start = idx * group_size
            end = min(start + group_size, N)
            # Add all indices in this group to the support
            support.extend(range(start, end))
        support = support[:K]  # Trim to exactly K indices if needed
    else:
        # Randomly select K indices
        support = np.random.choice(N, K, replace=False)

    # Set the non-zero coefficients with random complex values
    h[support] = (np.random.randn(len(support)) + 1j * np.random.randn(len(support))) / np.sqrt(2)

    return h, support


def generate_measurement_matrix(M, N):
    """
    Generate a measurement matrix with normalized columns.

    Parameters:
    M: Number of measurements (rows)
    N: Length of the channel vector (columns)

    Returns:
    Phi: The measurement matrix
    """
    # Generate a random complex matrix
    Phi = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)

    # Normalize columns
    for j in range(N):
        Phi[:, j] = Phi[:, j] / np.linalg.norm(Phi[:, j])

    return Phi


def add_noise(y, SNR_dB):
    """
    Add complex Gaussian noise to achieve a target SNR.

    Parameters:
    y: Noiseless measurements
    SNR_dB: Target signal-to-noise ratio in dB

    Returns:
    y_noisy: Noisy measurements
    """
    SNR_linear = 10 ** (SNR_dB / 10)
    signal_power = np.mean(np.abs(y) ** 2)
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))
    y_noisy = y + noise
    return y_noisy


def lasso_solver(y, Phi, lambda_val):
    """
    Solve the LASSO problem using scikit-learn.

    Parameters:
    y: Measurements
    Phi: Measurement matrix
    lambda_val: Regularization parameter

    Returns:
    h_est: Estimated channel vector
    """
    # Split real and imaginary parts to handle complex values
    real_y = np.real(y)
    imag_y = np.imag(y)
    real_Phi = np.real(Phi)
    imag_Phi = np.imag(Phi)

    # Construct augmented system for complex LASSO
    aug_y = np.concatenate([real_y, imag_y])
    aug_Phi = np.vstack([
        np.hstack([real_Phi, -imag_Phi]),
        np.hstack([imag_Phi, real_Phi])
    ])

    # Solve using scikit-learn's Lasso
    model = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=10000)
    model.fit(aug_Phi, aug_y)

    # Extract real and imaginary parts of the solution
    coeffs = model.coef_
    n_half = len(coeffs) // 2
    h_real = coeffs[:n_half]
    h_imag = coeffs[n_half:]

    # Combine to form complex solution
    h_est = h_real + 1j * h_imag

    return h_est


def group_lasso_solver(y, Phi, lambda_val, groups):
    """
    Solve the Group LASSO problem using a proximal gradient approach.

    Parameters:
    y: Measurements
    Phi: Measurement matrix
    lambda_val: Regularization parameter
    groups: List of lists, where each sublist contains indices for a group

    Returns:
    h_est: Estimated channel vector
    """
    max_iter = 1000
    tol = 1e-6
    N = Phi.shape[1]
    h = np.zeros(N, dtype=complex)

    # Precompute quantities
    PhiH_Phi = Phi.conj().T @ Phi
    PhiH_y = Phi.conj().T @ y

    # Compute Lipschitz constant of the gradient of the smooth part
    L = np.linalg.norm(PhiH_Phi, 2)
    step_size = 1.0 / L

    for it in range(max_iter):
        h_old = h.copy()

        # Gradient step
        grad = PhiH_Phi @ h - PhiH_y
        h_temp = h - step_size * grad

        # Proximal operator for group LASSO
        for group in groups:
            group_vec = h_temp[group]
            group_norm = np.linalg.norm(group_vec)

            if group_norm > step_size * lambda_val:
                h[group] = (1 - step_size * lambda_val / group_norm) * group_vec
            else:
                h[group] = 0

        # Check convergence
        if np.linalg.norm(h - h_old) < tol:
            break

    return h


def orthogonal_matching_pursuit(y, Phi, K):
    """
    Solve the sparse channel estimation problem using OMP.

    Parameters:
    y: Measurements
    Phi: Measurement matrix
    K: Sparsity level (number of non-zero coefficients to recover)

    Returns:
    h_est: Estimated channel vector
    support: Estimated support set
    """
    M, N = Phi.shape
    h_est = np.zeros(N, dtype=complex)
    support = []
    residual = y.copy()

    for _ in range(min(K, M)):  # Cannot recover more coefficients than measurements
        # Find the column most correlated with the residual
        correlation = np.abs(Phi.conj().T @ residual)
        idx = np.argmax(correlation)

        # Add the index to the support
        if idx not in support:
            support.append(idx)

        # Solve the least squares problem using the current support
        Phi_S = Phi[:, support]
        h_S = np.linalg.lstsq(Phi_S, y, rcond=None)[0]

        # Update the residual
        residual = y - Phi_S @ h_S

        # If residual is very small, break
        if np.linalg.norm(residual) < 1e-10:
            break

    # Update the solution
    h_est[support] = h_S

    return h_est, support


def calculate_nmse(h_true, h_est):
    """
    Calculate the Normalized Mean Square Error.

    Parameters:
    h_true: True channel vector
    h_est: Estimated channel vector

    Returns:
    nmse: Normalized Mean Square Error in dB
    """
    nmse = np.sum(np.abs(h_true - h_est) ** 2) / np.sum(np.abs(h_true) ** 2)
    nmse_db = 10 * np.log10(nmse)
    return nmse_db


def calculate_support_recovery_error(support_true, support_est):
    """
    Calculate the Support Recovery Error.

    Parameters:
    support_true: True support (indices of non-zero coefficients)
    support_est: Estimated support

    Returns:
    sre: Support Recovery Error
    """
    # Convert to sets for set operations
    true_set = set(support_true)
    est_set = set(support_est)

    # Calculate symmetric difference
    symmetric_diff = true_set.symmetric_difference(est_set)

    # Calculate SRE
    sre = len(symmetric_diff) / max(len(true_set), len(est_set))
    return sre


def create_groups(N, group_size):
    """
    Create groups of indices for Group LASSO.

    Parameters:
    N: Length of the channel vector
    group_size: Size of each group

    Returns:
    groups: List of lists, where each sublist contains indices for a group
    """
    groups = []
    for i in range(0, N, group_size):
        group = list(range(i, min(i + group_size, N)))
        groups.append(group)
    return groups


def identify_support(h_est, threshold=1e-6):
    """
    Identify the support of an estimated channel vector.

    Parameters:
    h_est: Estimated channel vector
    threshold: Threshold for considering a coefficient as non-zero

    Returns:
    support: Indices of coefficients above the threshold
    """
    return np.where(np.abs(h_est) > threshold)[0].tolist()


# -----------------------------------------------------------------------------
# Experiment 1: Performance vs SNR
# -----------------------------------------------------------------------------

def experiment_snr():
    # Parameters
    N = 200  # Length of channel vector
    M = 80  # Number of measurements
    K = 15  # Sparsity level
    SNR_range = np.linspace(0, 30, 7)  # SNR values in dB
    group_size = 4  # For Group LASSO
    num_trials = 50  # Number of Monte Carlo trials

    # Create groups for Group LASSO
    groups = create_groups(N, group_size)

    # Initialize results arrays
    nmse_lasso = np.zeros(len(SNR_range))
    nmse_group_lasso = np.zeros(len(SNR_range))
    nmse_omp = np.zeros(len(SNR_range))

    sre_lasso = np.zeros(len(SNR_range))
    sre_group_lasso = np.zeros(len(SNR_range))
    sre_omp = np.zeros(len(SNR_range))

    # Regularization parameter for LASSO and Group LASSO
    lambda_lasso = 0.01
    lambda_group_lasso = 0.01

    # Run Monte Carlo simulations
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")

        # Generate a sparse channel and measurement matrix
        h_true, support_true = generate_channel(N, K, is_group_sparse=True, group_size=group_size)
        Phi = generate_measurement_matrix(M, N)

        # Generate noiseless measurements
        y_noiseless = Phi @ h_true

        # Test different SNR values
        for i, snr in enumerate(SNR_range):
            # Add noise to achieve target SNR
            y = add_noise(y_noiseless, snr)

            # LASSO
            h_lasso = lasso_solver(y, Phi, lambda_lasso)
            support_lasso = identify_support(h_lasso)
            nmse_lasso[i] += calculate_nmse(h_true, h_lasso)
            sre_lasso[i] += calculate_support_recovery_error(support_true, support_lasso)

            # Group LASSO
            h_group_lasso = group_lasso_solver(y, Phi, lambda_group_lasso, groups)
            support_group_lasso = identify_support(h_group_lasso)
            nmse_group_lasso[i] += calculate_nmse(h_true, h_group_lasso)
            sre_group_lasso[i] += calculate_support_recovery_error(support_true, support_group_lasso)

            # OMP
            h_omp, support_omp = orthogonal_matching_pursuit(y, Phi, K)
            nmse_omp[i] += calculate_nmse(h_true, h_omp)
            sre_omp[i] += calculate_support_recovery_error(support_true, support_omp)

    # Average results
    nmse_lasso /= num_trials
    nmse_group_lasso /= num_trials
    nmse_omp /= num_trials

    sre_lasso /= num_trials
    sre_group_lasso /= num_trials
    sre_omp /= num_trials

    # Plot results
    plt.figure(figsize=(10, 8))

    # NMSE vs SNR
    plt.subplot(2, 1, 1)
    plt.plot(SNR_range, nmse_lasso, 'o-', label='LASSO')
    plt.plot(SNR_range, nmse_group_lasso, 's-', label='Group LASSO')
    plt.plot(SNR_range, nmse_omp, '^-', label='OMP')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title('Normalized Mean Square Error vs SNR')
    plt.legend()
    plt.grid(True)

    # SRE vs SNR
    plt.subplot(2, 1, 2)
    plt.plot(SNR_range, sre_lasso, 'o-', label='LASSO')
    plt.plot(SNR_range, sre_group_lasso, 's-', label='Group LASSO')
    plt.plot(SNR_range, sre_omp, '^-', label='OMP')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Support Recovery Error')
    plt.title('Support Recovery Error vs SNR')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('performance_vs_snr.png', dpi=300)
    plt.show()

    # Return results
    return {
        'SNR_range': SNR_range,
        'NMSE': {
            'LASSO': nmse_lasso,
            'Group LASSO': nmse_group_lasso,
            'OMP': nmse_omp
        },
        'SRE': {
            'LASSO': sre_lasso,
            'Group LASSO': sre_group_lasso,
            'OMP': sre_omp
        }
    }


# -----------------------------------------------------------------------------
# Experiment 2: Performance vs Sparsity Level
# -----------------------------------------------------------------------------

def experiment_sparsity():
    # Parameters
    N = 200  # Length of channel vector
    M = 80  # Number of measurements
    K_range = np.array([5, 10, 15, 20, 25, 30])  # Sparsity levels
    SNR = 20  # Fixed SNR in dB
    group_size = 4  # For Group LASSO
    num_trials = 50  # Number of Monte Carlo trials

    # Create groups for Group LASSO
    groups = create_groups(N, group_size)

    # Initialize results arrays
    nmse_lasso = np.zeros(len(K_range))
    nmse_group_lasso = np.zeros(len(K_range))
    nmse_omp = np.zeros(len(K_range))

    sre_lasso = np.zeros(len(K_range))
    sre_group_lasso = np.zeros(len(K_range))
    sre_omp = np.zeros(len(K_range))

    # Regularization parameter for LASSO and Group LASSO
    lambda_lasso = 0.01
    lambda_group_lasso = 0.01

    # Run Monte Carlo simulations
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")

        # Generate a measurement matrix
        Phi = generate_measurement_matrix(M, N)

        # Test different sparsity levels
        for i, K in enumerate(K_range):
            # Generate a sparse channel with given sparsity
            h_true, support_true = generate_channel(N, K, is_group_sparse=True, group_size=group_size)

            # Generate noiseless measurements
            y_noiseless = Phi @ h_true

            # Add noise to achieve target SNR
            y = add_noise(y_noiseless, SNR)

            # LASSO
            h_lasso = lasso_solver(y, Phi, lambda_lasso)
            support_lasso = identify_support(h_lasso)
            nmse_lasso[i] += calculate_nmse(h_true, h_lasso)
            sre_lasso[i] += calculate_support_recovery_error(support_true, support_lasso)

            # Group LASSO
            h_group_lasso = group_lasso_solver(y, Phi, lambda_group_lasso, groups)
            support_group_lasso = identify_support(h_group_lasso)
            nmse_group_lasso[i] += calculate_nmse(h_true, h_group_lasso)
            sre_group_lasso[i] += calculate_support_recovery_error(support_true, support_group_lasso)

            # OMP
            h_omp, support_omp = orthogonal_matching_pursuit(y, Phi, K)
            nmse_omp[i] += calculate_nmse(h_true, h_omp)
            sre_omp[i] += calculate_support_recovery_error(support_true, support_omp)

    # Average results
    nmse_lasso /= num_trials
    nmse_group_lasso /= num_trials
    nmse_omp /= num_trials

    sre_lasso /= num_trials
    sre_group_lasso /= num_trials
    sre_omp /= num_trials

    # Plot results
    plt.figure(figsize=(10, 8))

    # NMSE vs Sparsity
    plt.subplot(2, 1, 1)
    plt.plot(K_range, nmse_lasso, 'o-', label='LASSO')
    plt.plot(K_range, nmse_group_lasso, 's-', label='Group LASSO')
    plt.plot(K_range, nmse_omp, '^-', label='OMP')
    plt.xlabel('Sparsity Level (K)')
    plt.ylabel('NMSE (dB)')
    plt.title('Normalized Mean Square Error vs Sparsity Level')
    plt.legend()
    plt.grid(True)

    # SRE vs Sparsity
    plt.subplot(2, 1, 2)
    plt.plot(K_range, sre_lasso, 'o-', label='LASSO')
    plt.plot(K_range, sre_group_lasso, 's-', label='Group LASSO')
    plt.plot(K_range, sre_omp, '^-', label='OMP')
    plt.xlabel('Sparsity Level (K)')
    plt.ylabel('Support Recovery Error')
    plt.title('Support Recovery Error vs Sparsity Level')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('performance_vs_sparsity.png', dpi=300)
    plt.show()

    # Return results
    return {
        'K_range': K_range,
        'NMSE': {
            'LASSO': nmse_lasso,
            'Group LASSO': nmse_group_lasso,
            'OMP': nmse_omp
        },
        'SRE': {
            'LASSO': sre_lasso,
            'Group LASSO': sre_group_lasso,
            'OMP': sre_omp
        }
    }


# -----------------------------------------------------------------------------
# Experiment 3: Performance vs Regularization Parameter
# -----------------------------------------------------------------------------

def experiment_regularization():
    # Parameters
    N = 200  # Length of channel vector
    M = 80  # Number of measurements
    K = 15  # Sparsity level
    SNR = 20  # Fixed SNR in dB
    group_size = 4  # For Group LASSO
    num_trials = 30  # Number of Monte Carlo trials

    # Create groups for Group LASSO
    groups = create_groups(N, group_size)

    # Regularization parameter range
    lambda_range = np.logspace(-4, 0, 9)  # Log scale from 10^-4 to 10^0

    # Initialize results arrays
    nmse_lasso = np.zeros(len(lambda_range))
    nmse_group_lasso = np.zeros(len(lambda_range))

    sre_lasso = np.zeros(len(lambda_range))
    sre_group_lasso = np.zeros(len(lambda_range))

    # Run Monte Carlo simulations
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")

        # Generate a sparse channel and measurement matrix
        h_true, support_true = generate_channel(N, K, is_group_sparse=True, group_size=group_size)
        Phi = generate_measurement_matrix(M, N)

        # Generate noiseless measurements
        y_noiseless = Phi @ h_true

        # Add noise to achieve target SNR
        y = add_noise(y_noiseless, SNR)

        # OMP result (for reference)
        h_omp, support_omp = orthogonal_matching_pursuit(y, Phi, K)
        nmse_omp = calculate_nmse(h_true, h_omp)
        sre_omp = calculate_support_recovery_error(support_true, support_omp)

        # Test different regularization parameters
        for i, lambda_val in enumerate(lambda_range):
            # LASSO
            h_lasso = lasso_solver(y, Phi, lambda_val)
            support_lasso = identify_support(h_lasso)
            nmse_lasso[i] += calculate_nmse(h_true, h_lasso)
            sre_lasso[i] += calculate_support_recovery_error(support_true, support_lasso)

            # Group LASSO
            h_group_lasso = group_lasso_solver(y, Phi, lambda_val, groups)
            support_group_lasso = identify_support(h_group_lasso)
            nmse_group_lasso[i] += calculate_nmse(h_true, h_group_lasso)
            sre_group_lasso[i] += calculate_support_recovery_error(support_true, support_group_lasso)

    # Average results
    nmse_lasso /= num_trials
    nmse_group_lasso /= num_trials

    sre_lasso /= num_trials
    sre_group_lasso /= num_trials

    # Plot results
    plt.figure(figsize=(10, 8))

    # NMSE vs Regularization
    plt.subplot(2, 1, 1)
    plt.semilogx(lambda_range, nmse_lasso, 'o-', label='LASSO')
    plt.semilogx(lambda_range, nmse_group_lasso, 's-', label='Group LASSO')
    plt.axhline(y=nmse_omp, color='r', linestyle='--', label=f'OMP (K={K})')
    plt.xlabel('Regularization Parameter λ')
    plt.ylabel('NMSE (dB)')
    plt.title('Normalized Mean Square Error vs Regularization Parameter')
    plt.legend()
    plt.grid(True)

    # SRE vs Regularization
    plt.subplot(2, 1, 2)
    plt.semilogx(lambda_range, sre_lasso, 'o-', label='LASSO')
    plt.semilogx(lambda_range, sre_group_lasso, 's-', label='Group LASSO')
    plt.axhline(y=sre_omp, color='r', linestyle='--', label=f'OMP (K={K})')
    plt.xlabel('Regularization Parameter λ')
    plt.ylabel('Support Recovery Error')
    plt.title('Support Recovery Error vs Regularization Parameter')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('performance_vs_regularization.png', dpi=300)
    plt.show()

    # Return results
    return {
        'lambda_range': lambda_range,
        'NMSE': {
            'LASSO': nmse_lasso,
            'Group LASSO': nmse_group_lasso,
            'OMP': nmse_omp
        },
        'SRE': {
            'LASSO': sre_lasso,
            'Group LASSO': sre_group_lasso,
            'OMP': sre_omp
        }
    }


# -----------------------------------------------------------------------------
# Experiment 4: Performance vs Number of Measurements
# -----------------------------------------------------------------------------

def experiment_measurements():
    # Parameters
    N = 200  # Length of channel vector
    M_range = np.array([40, 60, 80, 100, 120, 140])  # Number of measurements
    K = 15  # Sparsity level
    SNR = 20  # Fixed SNR in dB
    group_size = 4  # For Group LASSO
    num_trials = 50  # Number of Monte Carlo trials

    # Regularization parameter for LASSO and Group LASSO
    lambda_lasso = 0.01
    lambda_group_lasso = 0.01

    # Initialize results arrays
    nmse_lasso = np.zeros(len(M_range))
    nmse_group_lasso = np.zeros(len(M_range))
    nmse_omp = np.zeros(len(M_range))

    sre_lasso = np.zeros(len(M_range))
    sre_group_lasso = np.zeros(len(M_range))
    sre_omp = np.zeros(len(M_range))

    # Run Monte Carlo simulations
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")

        # Generate a sparse channel
        h_true, support_true = generate_channel(N, K, is_group_sparse=True, group_size=group_size)

        # Test different numbers of measurements
        for i, M in enumerate(M_range):
            # Create groups for Group LASSO
            groups = create_groups(N, group_size)

            # Generate a measurement matrix
            Phi = generate_measurement_matrix(M, N)

            # Generate noiseless measurements
            y_noiseless = Phi @ h_true

            # Add noise to achieve target SNR
            y = add_noise(y_noiseless, SNR)

            # LASSO
            h_lasso = lasso_solver(y, Phi, lambda_lasso)
            support_lasso = identify_support(h_lasso)
            nmse_lasso[i] += calculate_nmse(h_true, h_lasso)
            sre_lasso[i] += calculate_support_recovery_error(support_true, support_lasso)

            # Group LASSO
            h_group_lasso = group_lasso_solver(y, Phi, lambda_group_lasso, groups)
            support_group_lasso = identify_support(h_group_lasso)
            nmse_group_lasso[i] += calculate_nmse(h_true, h_group_lasso)
            sre_group_lasso[i] += calculate_support_recovery_error(support_true, support_group_lasso)

            # OMP
            h_omp, support_omp = orthogonal_matching_pursuit(y, Phi, K)
            nmse_omp[i] += calculate_nmse(h_true, h_omp)
            sre_omp[i] += calculate_support_recovery_error(support_true, support_omp)

    # Average results
    nmse_lasso /= num_trials
    nmse_group_lasso /= num_trials
    nmse_omp /= num_trials

    sre_lasso /= num_trials
    sre_group_lasso /= num_trials
    sre_omp /= num_trials

    # Plot results
    plt.figure(figsize=(10, 8))

    # NMSE vs Measurements
    plt.subplot(2, 1, 1)
    plt.plot(M_range, nmse_lasso, 'o-', label='LASSO')
    plt.plot(M_range, nmse_group_lasso, 's-', label='Group LASSO')
    plt.plot(M_range, nmse_omp, '^-', label='OMP')
    plt.xlabel('Number of Measurements (M)')
    plt.ylabel('NMSE (dB)')