# %%
import numpy as np
from scipy.stats import poisson, gamma, rankdata, kendalltau
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Parameters
lambda_claims = 5  # Mean of Poisson distribution (number of claims)
alpha = 2.0        # Shape parameter of Gamma distribution (claim cost)
beta = 50.0        # Scale parameter of Gamma distribution (claim cost)
size = 1000        # Sample size

# Function to generate dependent uniform variables using a known copula
def generate_dependent_uniforms(size, copula_func, theta):
    U1 = np.random.uniform(size=size)
    U2 = copula_func(U1, theta)
    return U1, U2

# Gumbel copula function to generate dependent uniform variables
def gumbel_copula(U1, theta):
    W = -np.log(U1)
    Z = np.random.exponential(scale=1.0, size=len(U1))
    V = ((W**theta + Z**theta)**(1/theta))
    U2 = np.exp(-V)
    return U2

# Generate dependent uniform variables using a Gumbel copula
theta_true = 2.0
U_claims, U_costs = generate_dependent_uniforms(size, gumbel_copula, theta_true)

# Transform uniform variables to Poisson and Gamma variables
num_claims = poisson.ppf(U_claims, mu=lambda_claims)
claim_costs = gamma.ppf(U_costs, a=alpha, scale=beta)

# Filter out zero claims for empirical CDF and copula fitting
non_zero_indices = num_claims > -1
filtered_claims = num_claims[non_zero_indices]
filtered_costs = claim_costs[non_zero_indices]

# Function to calculate empirical CDF
def empirical_cdf(data):
    ranks = rankdata(data, method='ordinal')
    cdf_values = ranks / (len(data) + 1)
    return cdf_values

U_claims = empirical_cdf(filtered_claims)
U_costs = empirical_cdf(filtered_costs)

# Function to calculate the negative log-likelihood for the Gumbel copula
def gumbel_negative_log_likelihood(theta, U1, U2):
    if theta <= 1:
        return np.inf
    A = ((-np.log(U1))**theta + (-np.log(U2))**theta)**(1/theta)
    log_likelihood = -np.sum(A + (theta - 1) * np.log(U1 * U2) - (theta - 1) * A**(theta / (theta - 1)))
    return log_likelihood

# Function to estimate theta with multiple starting points
def estimate_gumbel_theta(U_claims, U_costs):
    initial_guesses = [1.1, 1.5, 2.0, 2.5, 3.0]  # Gumbel theta must be > 1
    best_theta = None
    best_nll = np.inf

    for initial_guess in initial_guesses:
        result = minimize(gumbel_negative_log_likelihood, x0=initial_guess, args=(U_claims, U_costs), bounds=[(1.01, None)])
        if result.fun < best_nll:
            best_nll = result.fun
            best_theta = result.x[0]
    
    return best_theta

# Estimate Gumbel copula parameter
theta_estimated = estimate_gumbel_theta(U_claims, U_costs)

print(f"True Gumbel Copula Parameter: {theta_true:.4f}")
print(f"Estimated Gumbel Copula Parameter: {theta_estimated:.4f}")

# Function to simulate joint scenarios using the estimated Gumbel copula parameter
def simulate_gumbel_copula_joint_poisson_gamma(lambda_claims, alpha, beta, theta, size):
    U1 = np.random.uniform(size=size)
    U2 = gumbel_copula(U1, theta)

    num_claims_simulated = poisson.ppf(U1, mu=lambda_claims)
    claim_costs_simulated = gamma.ppf(U2, a=alpha, scale=beta)

    return num_claims_simulated, claim_costs_simulated

# Simulate joint scenarios
num_claims_simulated, claim_costs_simulated = simulate_gumbel_copula_joint_poisson_gamma(lambda_claims, alpha, beta, theta_estimated, size)

# Total claim cost for each scenario
total_claim_costs = num_claims_simulated * claim_costs_simulated

# Risk measures: Value at Risk (VaR) and Expected Shortfall (ES)
def value_at_risk(returns, confidence_level):
    return np.percentile(returns, 100 * ( confidence_level))

def expected_shortfall(returns, confidence_level):
    var = value_at_risk(returns, confidence_level)
    return returns[returns >= var].mean()

confidence_level = 0.95
var_95 = value_at_risk(total_claim_costs, confidence_level)
es_95 = expected_shortfall(total_claim_costs, confidence_level)

print(f"95% VaR: ${var_95:.2f}")
print(f"95% Expected Shortfall: ${es_95:.2f}")

# Visualize the total claim costs
plt.figure(figsize=(10, 6))
plt.hist(total_claim_costs, bins=50, alpha=0.75, edgecolor='black')
plt.axvline(var_95, color='r', linestyle='dashed', linewidth=2, label=f'95% VaR = ${var_95:.2f}')
plt.axvline(es_95, color='b', linestyle='dashed', linewidth=2, label=f'95% ES = ${es_95:.2f}')
plt.title('Histogram of Total Claim Costs')
plt.xlabel('Total Claim Cost')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot to visualize the dependence structure
plt.figure(figsize=(10, 6))
plt.scatter(num_claims, claim_costs, alpha=0.5, edgecolor='k')
plt.title('Scatter Plot of Poisson and Gamma Variables')
plt.xlabel('Number of Claims (Poisson)')
plt.ylabel('Claim Costs (Gamma)')
plt.grid(True)
plt.show()
