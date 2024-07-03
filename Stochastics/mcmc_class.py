import numpy as np
import matplotlib.pyplot as plt

class BayesianModel:
    def __init__(self, likelihood_function, priors):
        """
        :param likelihood_function: A function that computes the likelihood given the data and parameters.
        :param priors: A dictionary of prior functions for each parameter.
        """
        self.likelihood_function = likelihood_function
        self.priors = priors

    def prior(self, params):
        """
        Compute the product of priors for the given parameters.
        :param params: A dictionary of parameters.
        :return: The product of prior probabilities.
        """
        prior_prob = 1
        for param, value in params.items():
            prior_prob *= self.priors[param](value)
        return prior_prob

    def posterior(self, data, params):
        """
        Compute the posterior probability for the given data and parameters.
        :param data: Observed data.
        :param params: A dictionary of parameters.
        :return: The posterior probability.
        """
        return self.likelihood_function(data, params) * self.prior(params)


class MCMCSampler:
    def __init__(self, model, data, initial_params, num_iterations=5000, burn_in=1000, proposal_sds=None):
        """
        :param model: An instance of the BayesianModel class.
        :param data: Observed data.
        :param initial_params: Initial values of the parameters as a dictionary.
        :param num_iterations: Number of MCMC iterations.
        :param burn_in: Number of burn-in iterations.
        :param proposal_sds: Standard deviations for the proposal distributions as a dictionary.
        """
        self.model = model
        self.data = data
        self.initial_params = initial_params
        self.num_iterations = num_iterations
        self.burn_in = burn_in
        self.proposal_sds = proposal_sds if proposal_sds is not None else {k: 0.5 for k in initial_params}
        self.samples = {k: [] for k in initial_params}

    def run(self):
        current_params = self.initial_params.copy()

        for _ in range(self.num_iterations):
            proposed_params = {
                k: np.random.normal(current_params[k], self.proposal_sds[k])
                for k in current_params
            }

            posterior_current = self.model.posterior(self.data, current_params)
            posterior_proposed = self.model.posterior(self.data, proposed_params)
            acceptance_ratio = posterior_proposed / posterior_current

            if np.random.rand() < acceptance_ratio:
                current_params = proposed_params

            for k in current_params:
                self.samples[k].append(current_params[k])

        for k in self.samples:
            self.samples[k] = np.array(self.samples[k][self.burn_in:])

    def get_samples(self):
        return self.samples


class ResultAnalyzer:
    def __init__(self, samples):
        """
        :param samples: A dictionary of MCMC samples.
        """
        self.samples = samples

    def plot_trace_and_histogram(self):
        num_params = len(self.samples)
        fig, ax = plt.subplots(num_params, 2, figsize=(12, 4 * num_params))

        if num_params == 1:
            ax = np.expand_dims(ax, axis=0)

        for i, (param, values) in enumerate(self.samples.items()):
            ax[i, 0].plot(values)
            ax[i, 0].set_title(f'Trace plot for {param}')
            ax[i, 0].set_xlabel('Iteration')
            ax[i, 0].set_ylabel(param)

            ax[i, 1].hist(values, bins=30, density=True, color='blue', edgecolor='black')
            ax[i, 1].set_title(f'Posterior distribution of {param}')
            ax[i, 1].set_xlabel(param)
            ax[i, 1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def estimate_parameters(self):
        estimates = {param: np.mean(values) for param, values in self.samples.items()}
        return estimates


def likelihood_function(data, params):
    mu = params['mu']
    sigma = params['sigma']
    if sigma <= 0:
        return 0
    return np.prod((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((data - mu) / sigma) ** 2))

def prior_mu(mu):
    return 1  # Uniform prior

def prior_sigma(sigma):
    return 1 if sigma > 0 else 0  # Uniform prior for sigma > 0

def main():
    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=2.0, size=100)
    initial_params = {'mu': np.mean(data), 'sigma': np.std(data)}
    priors = {'mu': prior_mu, 'sigma': prior_sigma}

    model = BayesianModel(likelihood_function, priors)
    sampler = MCMCSampler(model, data, initial_params, num_iterations=5000, burn_in=1000, proposal_sds={'mu': 0.5, 'sigma': 0.5})
    sampler.run()

    samples = sampler.get_samples()
    analyzer = ResultAnalyzer(samples)
    analyzer.plot_trace_and_histogram()

    estimates = analyzer.estimate_parameters()
    print(f'Estimated parameters: {estimates}')

if __name__ == "__main__":
    main()
