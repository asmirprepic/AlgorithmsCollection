import numpy as np
import matplotlib.pyplot as plt


class BayesianCIRCalibrator:
    def __init__(self, observed_data, dt, prior_mu, prior_sigma, likelihood_sigma):
        """
        Bayesian calibrator for the CIR process using Metropolis-Hastings.

        Parameters:
        - observed_data : array-like, observed variance process
        - dt : float, time step between observations
        - prior_mu : list, prior means for [kappa, theta, sigma]
        - prior_sigma : list, prior stddevs for [kappa, theta, sigma]
        - likelihood_sigma : float, assumed observational noise stddev
        """
        self.data = np.array(observed_data)
        self.dt = dt
        self.prior_mu = np.array(prior_mu)
        self.prior_sigma = np.array(prior_sigma)
        self.likelihood_sigma = likelihood_sigma
