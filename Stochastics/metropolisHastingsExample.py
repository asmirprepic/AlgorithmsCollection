# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 23:46:33 2023

@author: XQ966PY
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Set parameters
np.random.seed(12345)
B = 1

# Define the likelihood
def likelihood(y, A, B):
    return np.where(A>0,(B**A / gamma(A)) * y**(A-1) * np.exp(-B*y),0)

# Calculate and visualize the likelihood surface
yy = np.linspace(0, 10, 100)
AA = np.linspace(0.1, 5, 100)
likeSurf = np.zeros((len(yy), len(AA)))
for iA, A in enumerate(AA):
    likeSurf[:, iA] = likelihood(yy, A, B)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
AA_mesh, YY_mesh = np.meshgrid(AA, yy)
ax.plot_surface(AA_mesh, YY_mesh, likeSurf, cmap='hot')
ax.set_ylabel('p(y|A)')
ax.set_xlabel('A')
ax.set_title('p(y|A)')

# Display conditional at A = 2
ax.plot3D(np.ones_like(AA)*40, np.arange(1, 101), likeSurf[:, 40], 'g', linewidth=3)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_xticks([0, 100])
ax.set_xticklabels([0, 5])
ax.set_yticks([0, 100])
ax.set_yticklabels([0, 10])
ax.view_init(65, 25)
ax.legend(['p(y|A=2)'], loc='upper right')

# Define the prior
def prior(A):
    return np.sin(np.pi * A)**2

# Define the posterior
def posterior(y, A, B):
    return likelihood(y, A, B) * prior(A)

# Calculate and display the posterior surface
postSurf = np.zeros_like(likeSurf)
for iA, A in enumerate(AA):
    postSurf[:, iA] = posterior(yy, A, B)

zlim_min = np.min(postSurf[postSurf > 0])
zlim_max = np.max(postSurf[postSurf > 0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(AA_mesh, YY_mesh, postSurf, cmap='hot')
ax.set_ylabel('y')
ax.set_xlabel('A')
ax.set_title('p(A|y)')

# Display the prior
ax.plot3D(np.arange(1, 101), np.ones_like(AA)*100, prior(AA), 'b', linewidth=3)

# Sample from p(A | y = 1.5)
y = 1.5
target = postSurf[15, :]

# Display the posterior
ax.plot3D(np.arange(1, 101), np.ones_like(AA)*15, postSurf[15, :], 'm', linewidth=3)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_xticks([0, 100])
ax.set_xticklabels([0, 5])
ax.set_yticks([0, 100])
ax.set_yticklabels([0, 10])
ax.view_init(65, 25)
ax.legend(['p(A)', 'p(A|y=1.5)'], loc='upper right')
plt.show()
# Initialize the Metropolis-Hastings sampler
# Define the proposal density
def q(x, mu):
    return np.exp(-x / mu) / mu

mu = 5

# Display the target and proposal distributions
plt.figure()
th = plt.plot(AA, target, 'm', linewidth=2)
qh = plt.plot(AA, q(AA, mu), 'k', linewidth=2)
plt.legend([th[0], qh[0]], ['Target, p(A)', 'Proposal, q(A)'])
plt.xlabel('A')

# Constants
nSamples = 5000
burnIn = 500
minn = 0.1
maxx = 5

# Initialize the sampler
x = np.zeros(nSamples)
x[0] = mu
t = 0

# Run the Metropolis-Hastings sampler
while t < nSamples - 1:
    t += 1

    # Sample from the proposal
    xStar = np.random.exponential(mu)

    # Correction factor
    c = q(x[t - 1], mu) / q(xStar, mu)

    # Calculate the (corrected) acceptance ratio
    alpha = min([1, posterior(y, xStar, B) / posterior(y, x[t - 1], B) * c])

    # Accept or reject?
    u = np.random.rand()
    if u < alpha:
        x[t] = xStar
    else:
        x[t] = x[t - 1]

# Display the Markov chain path
plt.figure()
plt.subplot(211)
plt.plot(x[:t], np.arange(1, t + 1))
plt.hlines(burnIn, 0, maxx / 2, 'g--', linewidth=2)
plt.ylabel('t')
plt.xlabel('samples, A')
plt.gca().invert_yaxis()
plt.ylim([0, t])
plt.xlim([0, maxx])
plt.title('Markov Chain Path')

# Display the samples
plt.subplot(212)
nBins = 100
sampleBins = np.linspace(minn, maxx, nBins)
counts, _ = np.histogram(x[burnIn:], bins=sampleBins)
plt.bar(sampleBins[:-1], counts / sum(counts), width=(maxx - minn) / nBins, align='edge', color='k')
plt.xlabel('samples, A')
plt.ylabel('p(A | y)')
plt.title('Samples')
plt.xlim([0, 10])

# Overlay target distribution
plt.plot(AA, target / sum(target), 'm-', linewidth=2)
plt.legend(['Sampled Distribution', 'Target Posterior'])
plt.axis('tight')

plt.show()
