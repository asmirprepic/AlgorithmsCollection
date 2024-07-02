# %%
import numpy as np
import matplotlib.pyplot as plt

class BayesianModel:
    def __init__(self, data):
        self.data = data

    def prior_mu(self, mu):
        return 1  # Uniform prior for mu

    def prior_sigma(self, sigma):
        return 1 if sigma > 0 else 0  # Uniform prior for sigma > 0

    def likelihood(self, mu, sigma):
        return np.prod((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((self.data - mu) / sigma) ** 2))

    def posterior(self, mu, sigma):
        return self.likelihood(mu, sigma) * self.prior_mu(mu) * self.prior_sigma(sigma)


class MCMCSampler:
    def __init__(self, model, num_iterations=5000, burn_in=1000, proposal_mu_sd=0.5, proposal_sigma_sd=0.5):
        self.model = model
        self.num_iterations = num_iterations
        self.burn_in = burn_in
        self.proposal_mu_sd = proposal_mu_sd
        self.proposal_sigma_sd = proposal_sigma_sd
        self.mu_samples = []
        self.sigma_samples = []

    def run(self):
        mu_current = np.mean(self.model.data)
        sigma_current = np.std(self.model.data)

        for _ in range(self.num_iterations):
            mu_proposal = np.random.normal(mu_current, self.proposal_mu_sd)
            sigma_proposal = np.random.normal(sigma_current, self.proposal_sigma_sd)

            if sigma_proposal > 0:
                posterior_current = self.model.posterior(mu_current, sigma_current)
                posterior_proposal = self.model.posterior(mu_proposal, sigma_proposal)
                acceptance_ratio = posterior_proposal / posterior_current
            else:
                acceptance_ratio = 0

            if np.random.rand() < acceptance_ratio:
                mu_current = mu_proposal
                sigma_current = sigma_proposal

            self.mu_samples.append(mu_current)
            self.sigma_samples.append(sigma_current)

        self.mu_samples = np.array(self.mu_samples)[self.burn_in:]
        self.sigma_samples = np.array(self.sigma_samples)[self.burn_in:]

    def get_samples(self):
        return self.mu_samples, self.sigma_samples


class ResultAnalyzer:
    def __init__(self, mu_samples, sigma_samples):
        self.mu_samples = mu_samples
        self.sigma_samples = sigma_samples

    def plot_trace_and_histogram(self):
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].plot(self.mu_samples)
        ax[0, 0].set_title('Trace plot for mu')
        ax[0, 0].set_xlabel('Iteration')
        ax[0, 0].set_ylabel('mu')

        ax[0, 1].hist(self.mu_samples, bins=30, density=True, color='blue', edgecolor='black')
        ax[0, 1].set_title('Posterior distribution of mu')
        ax[0, 1].set_xlabel('mu')
        ax[0, 1].set_ylabel('Density')

        ax[1, 0].plot(self.sigma_samples)
        ax[1, 0].set_title('Trace plot for sigma')
        ax[1, 0].set_xlabel('Iteration')
        ax[1, 0].set_ylabel('sigma')

        ax[1, 1].hist(self.sigma_samples, bins=30, density=True, color='green', edgecolor='black')
        ax[1, 1].set_title('Posterior distribution of sigma')
        ax[1, 1].set_xlabel('sigma')
        ax[1, 1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def estimate_parameters(self):
        mu_estimate = np.mean(self.mu_samples)
        sigma_estimate = np.mean(self.sigma_samples)
        return mu_estimate, sigma_estimate


def main():
    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=2.0, size=100)

    model = BayesianModel(data)
    sampler = MCMCSampler(model)
    sampler.run()

    mu_samples, sigma_samples = sampler.get_samples()
    analyzer = ResultAnalyzer(mu_samples, sigma_samples)
    analyzer.plot_trace_and_histogram()

    mu_estimate, sigma_estimate = analyzer.estimate_parameters()
    print(f'Estimated mean (mu): {mu_estimate}')
    print(f'Estimated standard deviation (sigma): {sigma_estimate}')


if __name__ == "__main__":
    main()
