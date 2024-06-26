# %%

# Import necessary libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the logistic growth model
def logistic_growth(N, t, r, K):
    dNdt = r * N * (1 - N / K)
    return dNdt

# Function to solve the logistic growth differential equation
def solve_logistic_growth(t, N0, r, K):
    solution = odeint(logistic_growth, N0, t, args=(r, K))
    return solution.ravel()

# Function to generate synthetic observed data
def generate_synthetic_data(t, N0, r, K, noise_level=2, seed=42):
    np.random.seed(seed)
    true_data = solve_logistic_growth(t, N0, r, K)
    observed_data = true_data + np.random.normal(0, noise_level, size=len(true_data))
    return observed_data

# Define the log likelihood function
def log_likelihood(params, t, observed_data, N0):
    r, K = params
    model_data = solve_logistic_growth(t, N0, r, K)
    sigma = 2  # Assuming known constant noise level
    return -0.5 * np.sum((observed_data - model_data) ** 2 / sigma ** 2)

# Simple Metropolis-Hastings MCMC sampler
def run_mcmc_sampler(t, observed_data, N0, initial_guess, nsteps=5000, step_size=0.1):
    ndim = len(initial_guess)
    current_params = np.array(initial_guess)
    samples = np.zeros((nsteps, ndim))
    
    current_log_prob = log_likelihood(current_params, t, observed_data, N0)
    
    for i in range(nsteps):
        new_params = current_params + step_size * np.random.randn(ndim)
        if np.any(new_params <= 0):  # Ensure parameters are positive
            samples[i] = current_params
            continue
        
        new_log_prob = log_likelihood(new_params, t, observed_data, N0)
        
        if new_log_prob > current_log_prob or np.random.rand() < np.exp(new_log_prob - current_log_prob):
            current_params = new_params
            current_log_prob = new_log_prob
        
        samples[i] = current_params
    
    return samples

# Function to analyze and plot the results
def analyze_results(samples, true_params):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axs[0].plot(samples[:, 0], label='r')
    axs[0].axhline(true_params[0], color='r', linestyle='--', label='true r')
    axs[0].set_ylabel('r')
    axs[0].legend()
    
    axs[1].plot(samples[:, 1], label='K')
    axs[1].axhline(true_params[1], color='r', linestyle='--', label='true K')
    axs[1].set_ylabel('K')
    axs[1].legend()
    
    plt.xlabel('Step')
    plt.show()

    # Plot histograms of the parameter estimates
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].hist(samples[:, 0], bins=30, density=True)
    axs[0].axvline(true_params[0], color='r', linestyle='--', label='true r')
    axs[0].set_xlabel('r')
    axs[0].set_ylabel('Density')
    axs[0].legend()

    axs[1].hist(samples[:, 1], bins=30, density=True)
    axs[1].axvline(true_params[1], color='r', linestyle='--', label='true K')
    axs[1].set_xlabel('K')
    axs[1].set_ylabel('Density')
    axs[1].legend()

    plt.show()

# Main function to perform the whole process
def main():
    # True parameters for the synthetic data
    r_true = 0.5
    K_true = 50
    N0_true = 10

    # Time points for the data
    t = np.linspace(0, 10, 1000)

    # Generate synthetic observed data
    observed_data = generate_synthetic_data(t, N0_true, r_true, K_true)

    # Initial guess for the parameters
    initial_guess = np.array([0.4, 45])

    # Run the MCMC sampler
    samples = run_mcmc_sampler(t, observed_data, N0_true, initial_guess)

    # Analyze and plot the results
    analyze_results(samples, [r_true, K_true])

# Run the main function
if __name__ == "__main__":
    main()
