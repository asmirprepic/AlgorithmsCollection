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

    def cir_model(self, params):
        kappa, theta, sigma = params
        v_model = np.zeros_like(self.data)
        v_model[0] = self.data[0]
        for i in range(1, len(self.data)):
            v_prev = v_model[i - 1]
            drift = kappa * (theta - v_prev) * self.dt
            diffusion = 0
            v_model[i] = v_prev + drift + diffusion
        return v_model

    def log_prior(self, params):
        return np.sum(norm.logpdf(params, self.prior_mu, self.prior_sigma))

    def log_likelihood(self, params):
        model_vals = self.cir_model(params)
        return np.sum(norm.logpdf(self.data, model_vals, self.likelihood_sigma))

    def log_posterior(self, params):
        return self.log_prior(params) + self.log_likelihood(params)

    def metropolis_hastings(self, initial_params, n_samples=10000, proposal_width=0.02):
        dim = len(initial_params)
        samples = np.zeros((n_samples, dim))
        current = np.array(initial_params)
        current_log_post = self.log_posterior(current)

        accepted = 0

        for i in range(n_samples):
            proposal = current + np.random.normal(0, proposal_width, size=dim)
            if np.any(proposal <= 0):
                samples[i] = current
                continue

            proposal_log_post = self.log_posterior(proposal)
            acceptance_ratio = np.exp(proposal_log_post - current_log_post)

            if np.random.rand() < acceptance_ratio:
                current = proposal
                current_log_post = proposal_log_post
                accepted += 1

            samples[i] = current

        return samples, accepted / n_samples

def simulate_cir_path(kappa, theta, sigma, v0, T, dt, seed=42):
    np.random.seed(seed)
    N = int(T / dt)
    v = np.zeros(N + 1)
    v[0] = v0
    for i in range(N):
        sqrt_v = np.sqrt(max(v[i], 0))
        v[i+1] = (v[i] + kappa*(theta - v[i])*dt +
                  sigma*sqrt_v*np.random.normal(0, np.sqrt(dt)))
    return v
