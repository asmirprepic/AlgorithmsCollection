import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, t
from scipy.optimize import differential_evolution

np.random.seed(42)

def generate_returns(n_samples=1000):
    df = 3
    returns = t.rvs(df, size=n_samples)
    return returns

def get_exceedances(data, threshold):
    exceedances = data[data > threshold] - threshold
    return exceedances

def gpd_neg_log_likelihood(params, data):
    xi, sigma = params
    if sigma <= 0 or np.any((1 + xi * data / sigma) <= 0):
        return np.inf
    n = len(data)
    log_likelihood = -n * np.log(sigma) - (1 + 1/xi) * np.sum(np.log(1 + xi * data / sigma))
    return -log_likelihood

def fit_gpd(exceedances, bounds=[(-0.5, 2.0), (0.01, 10.0)]):
    result = differential_evolution(gpd_neg_log_likelihood, bounds, args=(exceedances,), seed=42)
    if result.success:
        xi, sigma = result.x
        return xi, sigma
    else:
        raise ValueError("GPD fitting failed")

def compute_var(exceedances, xi, sigma, threshold, prob_exceed, alpha=0.95):
    u = threshold
    var = u + (sigma / xi) * ((prob_exceed / (1 - alpha)) ** (-xi) - 1)
    return var

def hill_estimator(exceedances):
    sorted_exceedances = np.sort(exceedances)[::-1]
    k = int(len(exceedances) * 0.1)  # Top 10% for Hill estimator
    if k < 2:
        return np.nan
    return 1 / (np.mean(np.log(sorted_exceedances[:k] / sorted_exceedances[k-1])))

if __name__ == "__main__":
    data = generate_returns(10000)
    base_quantile = np.percentile(data, 95)
    threshold = base_quantile + np.random.uniform(-0.05, 0.05) * base_quantile
    exceedances = get_exceedances(data, threshold)
    xi, sigma = fit_gpd(exceedances)
    hill_xi = hill_estimator(exceedances)
    print(f"Fitted GPD parameters: xi={xi:.3f}, sigma={sigma:.3f}")
    print(f"Hill estimator for tail index: {hill_xi:.3f}")
    prob_exceed = len(exceedances) / len(data)
    var_95 = compute_var(exceedances, xi, sigma, threshold, prob_exceed, alpha=0.95)
    print(f"VaR at 95% confidence: {var_95:.3f}")
    x = np.linspace(0, max(exceedances), 100)
    gpd_pdf = genpareto.pdf(x, xi, scale=sigma)
    plt.hist(exceedances, bins=30, density=True, alpha=0.5, color='purple', label="Empirical Exceedances")
    plt.plot(x, gpd_pdf, 'orange', label=f"GPD Fit (xi={xi:.2f}, sigma={sigma:.2f})")
    plt.axvline(var_95 - threshold, color='cyan', linestyle='--', label=f"VaR 95% = {var_95:.2f}")
    plt.yscale('log')
    plt.ylim(1e-3, plt.ylim()[1])
    plt.title("GPD Fit to Exceedances (Log Scale)")
    plt.xlabel("Exceedance Value")
    plt.ylabel("Density (Log)")
    plt.legend()
    plt.show()
