import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class BayesianCalibrator:
    def __init__(self, model_func,data,prior_mu,prior_sigma,likelihood_sigma):
        """
        Bayesian Calibrator using Metropolis-Hastings.

        Parameters:
        - model_func: function taking parameters and returning predictions
        - data: observed data to calibrate against
        - prior_mu: prior mean for parameters
        - prior_sigma: prior stddev for parameters
        - likelihood_sigma: noise stddev in the likelihood model
        """
        self.model_func = model_func
        self.data = data
        self.prior_mu = np.array(prior_mu)
        self.prior_sigma = np.array(prior_sigma)
        self.likelihood_sigma = likelihood_sigma
