import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x):
    return 0.3 * np.exp(-0.2 * (x - 2)**2) + 0.7 * np.exp(-0.2 * (x + 2)**2)

def metropolis_hastings(num_samples, proposal_width):
    samples = np.zeros(num_samples)
    samples[0] = np.random.rand()
    
    for i in range(1, num_samples):
        current_sample = samples[i-1]
        proposal = np.random.normal(current_sample, proposal_width)
        
        acceptance_ratio = target_distribution(proposal) / target_distribution(current_sample)
        
        if np.random.rand() < acceptance_ratio:
            samples[i] = proposal
        else:
            samples[i] = current_sample
            
    return samples

def main():
    num_samples = 10000
    proposal_width = 1.0
    
    samples = metropolis_hastings(num_samples, proposal_width)
    
    plt.figure(figsize=(10, 6))
    x = np.linspace(-10, 10, 1000)
    y = target_distribution(x)
    plt.plot(x, y, label='Target Distribution')
    plt.hist(samples, bins=50, density=True, alpha=0.5, label='MCMC Samples')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Metropolis-Hastings MCMC')
    plt.show()
    
    print(f"Estimated mean: {np.mean(samples):.4f}")
    
if __name__ == "__main__":
    main()
