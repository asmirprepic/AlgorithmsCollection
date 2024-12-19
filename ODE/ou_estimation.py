from scipy.optimize import curve_fit

def ou_autocorrelation(lags, theta, sigma):
    """
    Theoretical autocorrelation for Ornstein-Uhlenbeck process.
    """
    return sigma**2 / (2 * theta) * np.exp(-theta * lags)

# Fit the autocorrelation function to estimate theta and sigma
def estimate_ou_parameters(P, dt):
    diffs = np.diff(P) / dt
    autocorr = np.correlate(P - np.mean(P), P - np.mean(P), mode="full") / len(P)
    autocorr = autocorr[len(P)-1:]  # Take positive lags
    lags = np.arange(len(autocorr)) * dt
    popt, _ = curve_fit(ou_autocorrelation, lags, autocorr, p0=[0.1, 5.0])
    theta_est, sigma_est = popt
    mu_est = np.mean(P)
    return theta_est, mu_est, sigma_est
