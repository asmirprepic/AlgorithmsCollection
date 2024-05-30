# %% 

import numpy as np
from scipy.stats import poisson, gamma
from scipy.stats import rankdata
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

# Parameters
lambda_claims = 5  # Mean of Poisson distribution (number of claims)
alpha = 2.0        # Shape parameter of Gamma distribution (claim cost)
beta = 50.0        # Scale parameter of Gamma distribution (claim cost)
size = 1000        # Sample size

# Simulate Poisson-distributed number of claims
num_claims = poisson.rvs(mu=lambda_claims, size=size)

# Simulate Gamma-distributed claim costs
claim_costs = gamma.rvs(a=alpha, scale=beta, size=size)




def empirical_cdf(data):
    ranks = rankdata(data, method='ordinal')
    cdf_values = ranks / (len(data) + 1)
    return cdf_values

U_claims = empirical_cdf(num_claims)
U_costs = empirical_cdf(claim_costs)


def estimate_clayton_theta(U1, U2):
    tau, _ = kendalltau(U1, U2)
    theta = 2 * tau / (1 - tau)
    return theta

theta_estimated = estimate_clayton_theta(U_claims, U_costs)
print(f"Estimated Clayton Copula Parameter: {theta_estimated:.4f}")


def simulate_clayton_copula_joint_poisson_gamma(lambda_claims, alpha, beta, theta, size):
    U1 = np.random.uniform(size=size)
    U2 = np.random.uniform(size=size)

    V = (U2**(-theta) - 1) * (U1**(-theta) - 1) + 1
    U2_dependent = V**(-1/theta)

    num_claims_simulated = poisson.ppf(U1, mu=lambda_claims)
    claim_costs_simulated = gamma.ppf(U2_dependent, a=alpha, scale=beta)

    return num_claims_simulated, claim_costs_simulated

# Simulate joint scenarios
num_claims_simulated, claim_costs_simulated = simulate_clayton_copula_joint_poisson_gamma(lambda_claims, alpha, beta, theta_estimated, size)


# Total claim cost for each scenario
total_claim_costs = num_claims_simulated * claim_costs_simulated

# Risk measures: Value at Risk (VaR) and Expected Shortfall (ES)
def value_at_risk(returns, confidence_level):
    return np.percentile(returns, 100 * (1 - confidence_level))

def expected_shortfall(returns, confidence_level):
    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()

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

# %%
