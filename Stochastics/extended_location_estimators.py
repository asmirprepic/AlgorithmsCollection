from scipy.stats import trim_mean
from scipy.stats.mstats import winsorize

def simulate_extended_location_estimators(contamination):
    true_value = 0
    mean_estimates, median_estimates = [], []
    trimmed_estimates, winsorized_estimates = [], []
    
    for _ in range(n_sim):
        x = np.random.normal(loc=true_value, scale=1, size=n)
        n_contaminated = int(contamination * n)
        if n_contaminated > 0:
            x[:n_contaminated] = 10  # large outliers
        
        mean_estimates.append(np.mean(x))
        median_estimates.append(np.median(x))
        trimmed_estimates.append(trim_mean(x, proportiontocut=0.1))
        winsorized_x = winsorize(x, limits=0.1)  # top and bottom 10%
        winsorized_estimates.append(np.mean(winsorized_x))
    
    def mse(estimates):
        return np.mean((np.array(estimates) - true_value) ** 2)

    return {
        "contamination": contamination,
        "mean_mse": mse(mean_estimates),
        "median_mse": mse(median_estimates),
        "trimmed_mse": mse(trimmed_estimates),
        "winsorized_mse": mse(winsorized_estimates),
        "mean_bias": np.mean(mean_estimates) - true_value,
        "median_bias": np.mean(median_estimates) - true_value,
        "trimmed_bias": np.mean(trimmed_estimates) - true_value,
        "winsorized_bias": np.mean(winsorized_estimates) - true_value
    }
