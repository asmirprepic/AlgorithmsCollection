import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def nelson_siegel_model(t, beta0, beta1, beta2, tau):
    """
    Nelson-Siegel model function.
    :param t: array of maturities
    :param beta0: level parameter
    :param beta1: slope parameter
    :param beta2: curvature parameter
    :param tau: scale parameter
    :return: array of yields
    """
    return beta0 + (beta1 + beta2) * (tau / t) * (1 - np.exp(-t / tau)) - beta2 * np.exp(-t / tau)

def fit_nelson_siegel(maturities, yields):
    """
    Fit the Nelson-Siegel model to the given data.
    :param maturities: array of maturities
    :param yields: array of observed yields
    :return: optimized parameters
    """
    initial_guess = [0.03, -0.02, 0.02, 1.0]
    result = optimize.minimize(lambda params: np.sum((yields - nelson_siegel_model(maturities, *params)) ** 2),
                               initial_guess, method='L-BFGS-B')
    return result.x

# Example usage
maturities = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])  # in years
observed_yields = np.array([0.02, 0.021, 0.025, 0.027, 0.03, 0.032, 0.035, 0.04, 0.045])  # example data

params = fit_nelson_siegel(maturities, observed_yields)
print("Fitted parameters:", params)

# Plotting the results
fitted_yields = nelson_siegel_model(maturities, *params)
plt.plot(maturities, observed_yields, 'o', label='Observed Yields')
plt.plot(maturities, fitted_yields, label='Fitted Curve', linestyle='--')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield')
plt.title('Nelson-Siegel Model Fit')
plt.legend()
plt.show()
