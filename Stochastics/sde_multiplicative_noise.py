# Parameters for the SDE
mu = lambda x, t: 0.1 * x        # Drift function
sigma = lambda x, t: 0.2 * x     # Diffusion function (multiplicative noise)
dsigma_dx = lambda x, t: 0.2     # Derivative of sigma with respect to x

# Initialize an array to store the paths
X = np.zeros((num_paths, N))
X[:, 0] = x0

# Generate paths using Milstein's scheme
for i in range(1, N):
    # Generate standard normal random variables for the Wiener process increments
    dW = np.sqrt(dt) * np.random.randn(num_paths)
    # Milstein update
    X[:, i] = (X[:, i-1] + mu(X[:, i-1], t[i-1]) * dt 
               + sigma(X[:, i-1], t[i-1]) * dW 
               + 0.5 * sigma(X[:, i-1], t[i-1]) * dsigma_dx(X[:, i-1], t[i-1]) * (dW**2 - dt))

# Plot the simulated paths
plt.figure(figsize=(10, 6))
for j in range(num_paths):
    plt.plot(t, X[j, :], lw=1.5)
plt.title('Simulated Paths of SDE with Multiplicative Noise (Milstein Method)')
plt.xlabel('Time t')
plt.ylabel('X(t)')
plt.grid(True)
plt.show()
