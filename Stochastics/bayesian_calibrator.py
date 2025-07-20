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

    def log_prior(self,params):
        return np.sum(norm.logpdf(params,self.prior_mu,self.prior_sigma))

    def log_likelihood(self,params):
        prediction = self.model_func(params)
        return np.sum(norm.logpdf(self.data,prediction,self.likelihood_sigma))

    def log_posterior(self,params):
        return self.log_prior(params) + self.log_likelihood(params)

    def metropolis_hastings(self, initial_params,n_samples = 10_000,proposal_width = 0.1):
        dim = len(initial_params)
        samples = np.zeros((n_samples,dim))
        current = np.array(initial_params)
        current_log_post = self.log_posterior(current)

        accepted = 0

        for i in len(samples):
            proposal = current + np.random.normal(0,proposal_width,size = dim)
            proposal_log_post = self.log_posterior(proposal)
            acceptance_ratio = np.exp(proposal_log_post-current_log_post)

            if np.random.rand() < acceptance_ratio:
                current = proposal
                current_log_post = proposal_log_post
                accepted += 1

            samples[i]

        acceptance_rate = accepted/n_samples
        return samples, acceptance_rate

# Example

np.random.seed(42)
x_data = np.linspace(0,10,50)
true_params = [2.0,1.0]
noise_sigma = 1.0

y_data = true_params[0]*x_data + true_params[1] + np.random.normal(0,noise_sigma,size = x_data)

def linear_model(params):
    a,b = params
    return a * x_data + b

calibrator = BayesianCalibrator(
    model_func=linear_model,
    prior_mu=[0.0,0.0],
    prior_sigma=[5.0,5.0]
    likelihood_sigma=noise_sigma
    )

samples,acc_rate = calibrator.metropolis_hastings(initial_params=[1.0,1.0],n_samples = 5000,proposal_width=0.1)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(samples[:, 0], bins=50, alpha=0.7)
axs[0].set_title("Posterior of a (slope)")
axs[1].hist(samples[:, 1], bins=50, alpha=0.7)
axs[1].set_title("Posterior of b (intercept)")
plt.suptitle(f"Metropolis-Hastings Posterior Samples (Acceptance Rate: {acc_rate:.2f})")
plt.show()
