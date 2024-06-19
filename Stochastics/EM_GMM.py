import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate synthetic data from a mixture of Gaussians
np.random.seed(42)
n_samples = 300
weights_true = np.array([0.4, 0.6])
means_true = np.array([0, 5])
std_devs_true = np.array([1, 1.5])

# Sample data
data = np.hstack([
    np.random.normal(means_true[i], std_devs_true[i], int(weights_true[i] * n_samples))
    for i in range(len(weights_true))
])

# Define the Gaussian Mixture Model
class GaussianMixtureModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.weights = np.ones(n_components) / n_components
        self.means = np.random.uniform(min(data), max(data), n_components)
        self.variances = np.ones(n_components)
        
    def e_step(self, data):
        likelihood = np.array([
            self.weights[k] * norm.pdf(data, self.means[k], np.sqrt(self.variances[k]))
            for k in range(self.n_components)
        ])
        likelihood_sum = np.sum(likelihood, axis=0)
        responsibilities = likelihood / likelihood_sum
        return responsibilities.T
    
    def m_step(self, data, responsibilities):
        n_k = np.sum(responsibilities, axis=0)
        self.weights = n_k / len(data)
        self.means = np.sum(responsibilities * data[:, np.newaxis], axis=0) / n_k
        self.variances = np.sum(responsibilities * (data[:, np.newaxis] - self.means) ** 2, axis=0) / n_k
    
    def log_likelihood(self, data):
        likelihood = np.array([
            self.weights[k] * norm.pdf(data, self.means[k], np.sqrt(self.variances[k]))
            for k in range(self.n_components)
        ])
        return np.sum(np.log(np.sum(likelihood, axis=0)))
    
    def fit(self, data, tol=1e-6, max_iter=100):
        log_likelihoods = []
        for i in range(max_iter):
            responsibilities = self.e_step(data)
            self.m_step(data, responsibilities)
            log_likelihood = self.log_likelihood(data)
            log_likelihoods.append(log_likelihood)
            if i > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                break
        return log_likelihoods

# Fit GMM to data
gmm = GaussianMixtureModel(n_components=2)
log_likelihoods = gmm.fit(data)

# Plot the log-likelihood progression
plt.figure(figsize=(12, 6))
plt.plot(log_likelihoods, marker='o')
plt.title('Log-Likelihood Progression during EM Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.grid(True)
plt.show()

# Plot the data and fitted Gaussians
x_range = np.linspace(min(data) - 2, max(data) + 2, 1000)
fitted_pdf = np.sum([
    gmm.weights[k] * norm.pdf(x_range, gmm.means[k], np.sqrt(gmm.variances[k]))
    for k in range(gmm.n_components)
], axis=0)

plt.figure(figsize=(12, 6))
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
plt.plot(x_range, fitted_pdf, color='red', lw=2, label='Fitted Gaussian Mixture')
for k in range(gmm.n_components):
    plt.plot(x_range, gmm.weights[k] * norm.pdf(x_range, gmm.means[k], np.sqrt(gmm.variances[k])), lw=2, linestyle='--', label=f'Component {k+1}')
plt.title('Gaussian Mixture Model Fit')
plt.xlabel('Data')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Print the final parameters
print("Estimated weights:", gmm.weights)
print("Estimated means:", gmm.means)
print("Estimated variances:", gmm.variances)
