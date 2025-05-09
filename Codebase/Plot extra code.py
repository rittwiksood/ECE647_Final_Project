import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


def generate_sparse_channel(Nt, Nr, S):
    """Generate sparse MIMO channel matrix without SNR normalization"""
    H = np.zeros((Nt, Nr), dtype=complex)
    nonzero_indices = np.random.choice(Nt * Nr, S, replace=False)
    values = (np.random.randn(S) + 1j * np.random.randn(S)) / np.sqrt(2)
    H.flat[nonzero_indices] = values
    return H / np.linalg.norm(H, 'fro') * np.sqrt(Nt * Nr)


def lasso_channel_estimation(y, Phi, lambda_lasso):
    """LASSO-based channel estimation"""
    P, Nr = y.shape
    Nt = Phi.shape[1]
    H_est = np.zeros((Nt, Nr), dtype=complex)

    for i in range(Nr):
        h = cp.Variable(Nt, complex=True)
        objective = cp.Minimize(cp.sum_squares(y[:, i] - Phi @ h) + lambda_lasso * cp.norm1(h))
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS)
        H_est[:, i] = h.value

    return H_est


def group_lasso_channel_estimation(y, Phi, lambda_glasso, group_size):
    """Group LASSO-based channel estimation"""
    P, Nr = y.shape
    Nt = Phi.shape[1]
    num_groups = Nt // group_size
    H_est = np.zeros((Nt, Nr), dtype=complex)

    for i in range(Nr):
        h = cp.Variable(Nt, complex=True)
        group_norms = [cp.norm2(h[g * group_size:(g + 1) * group_size]) for g in range(num_groups)]
        objective = cp.Minimize(cp.sum_squares(y[:, i] - Phi @ h) + lambda_glasso * cp.sum(group_norms))
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS)
        H_est[:, i] = h.value

    return H_est


def omp_channel_estimation(y, Phi, sparsity):
    """Orthogonal Matching Pursuit for channel estimation"""
    P, Nr = y.shape
    Nt = Phi.shape[1]
    H_est = np.zeros((Nt, Nr), dtype=complex)

    for i in range(Nr):
        residual = y[:, i].copy()
        idx_set = []
        Phi_r = Phi.copy()

        for _ in range(sparsity):
            new_idx = np.argmax(np.abs(Phi_r.T.conj() @ residual))
            idx_set.append(new_idx)
            Phi_active = Phi[:, idx_set]
            h_ls = np.linalg.pinv(Phi_active) @ y[:, i]
            residual = y[:, i] - Phi_active @ h_ls
            Phi_r[:, new_idx] = 0

        h_est = np.zeros(Nt, dtype=complex)
        h_est[idx_set] = h_ls
        H_est[:, i] = h_est

    return H_est


def evaluate_performance(H_true, H_est):
    """Compute NMSE and support recovery error"""
    nmse = np.linalg.norm(H_est - H_true, 'fro') ** 2 / np.linalg.norm(H_true, 'fro') ** 2
    true_support = np.abs(H_true) > 1e-3
    est_support = np.abs(H_est) > 1e-3
    support_error = np.mean(true_support != est_support)
    return nmse, support_error


def run_simulation(snr_db, Nt, Nr, S, P, lambda_lasso, lambda_glasso, group_size, num_trials):
    """Run simulation for a given SNR"""
    nmse_results = np.zeros((3, num_trials))  # LASSO, Group LASSO, OMP
    support_results = np.zeros((3, num_trials))

    Phi = (np.random.randn(P, Nt) + 1j * np.random.randn(P, Nt)) / np.sqrt(2 * P)
    sigma2 = 10 ** (-snr_db / 10)

    for trial in range(num_trials):
        H_true = generate_sparse_channel(Nt, Nr, S)
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(P, Nr) + 1j * np.random.randn(P, Nr))
        y = Phi @ H_true + noise

        # LASSO estimation
        H_lasso = lasso_channel_estimation(y, Phi, lambda_lasso)
        nmse_results[0, trial], support_results[0, trial] = evaluate_performance(H_true, H_lasso)

        # Group LASSO estimation
        H_glasso = group_lasso_channel_estimation(y, Phi, lambda_glasso, group_size)
        nmse_results[1, trial], support_results[1, trial] = evaluate_performance(H_true, H_glasso)

        # OMP estimation
        H_omp = omp_channel_estimation(y, Phi, S)
        nmse_results[2, trial], support_results[2, trial] = evaluate_performance(H_true, H_omp)

    return np.mean(nmse_results, axis=1), np.mean(support_results, axis=1)


# Main simulation parameters
Nt, Nr, S, P = 64, 8, 8, 16
lambda_lasso, lambda_glasso, group_size = 0.1, 0.1, 4
num_trials = 50
snr_range = np.arange(0, 31, 5)  # SNR from 0dB to 30dB in 5dB steps

# Store results
nmse_avg = np.zeros((len(snr_range), 3))
support_avg = np.zeros((len(snr_range), 3))

# Run simulations across SNR range
print("Starting simulations...")
for i, snr in enumerate(snr_range):
    nmse_avg[i], support_avg[i] = run_simulation(snr, Nt, Nr, S, P,
                                                 lambda_lasso, lambda_glasso,
                                                 group_size, num_trials)
    print(f"Completed SNR = {snr} dB")

# Plot results
plt.figure(figsize=(14, 6))

# NMSE Plot
plt.subplot(1, 2, 1)
plt.semilogy(snr_range, nmse_avg[:, 0], 'b-o', label='LASSO', linewidth=2, markersize=8)
plt.semilogy(snr_range, nmse_avg[:, 1], 'r-s', label='Group LASSO', linewidth=2, markersize=8)
plt.semilogy(snr_range, nmse_avg[:, 2], 'g-^', label='OMP', linewidth=2, markersize=8)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Normalized MSE', fontsize=12)
plt.title('Channel Estimation NMSE', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=12)

# Support Recovery Plot
plt.subplot(1, 2, 2)
plt.semilogy(snr_range, support_avg[:, 0], 'b-o', label='LASSO', linewidth=2, markersize=8)
plt.semilogy(snr_range, support_avg[:, 1], 'r-s', label='Group LASSO', linewidth=2, markersize=8)
plt.semilogy(snr_range, support_avg[:, 2], 'g-^', label='OMP', linewidth=2, markersize=8)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Support Recovery Error', fontsize=12)
plt.title('Sparsity Pattern Recovery', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()