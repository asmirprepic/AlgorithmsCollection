import numpy as np
import matplotlib.pyplot as plt

def simulate_ou_lambda(t_grid,theta =2,mu=15,sigma= 3, lam0 = 5):
    dt = t_grid[1]-t_grid[0]
    lam = np.zeros_like(t_grid)
    lam[0] = lam0
    for i in range(1,len(t_grid)):
        dW = np.random.normal(0, np.sqrt(dt))
        lam[i] = lam[i-1] + theta * (mu - lam[i-1])*dt + sigma*dW
        lam[i] = max(lam[i],0)
    return lam

def simulate_cox_process(lam_t,t_grid):
    lam_max = np.max(lam_t)
    events = []
    t = t_grid[0]
    while t < t_grid[-1]:
        u = np.random.exponential(scale=1/lam_max)
        t += u
        if t >= t_grid[-1]:
            break
        lam_interp = np.interp(t,t_grid,lam_t)
        if np.random.rand() < lam_interp / lam_max:
            events.append(t)
    return events


t_grid = np.linspace(0, 9, 1000)
np.random.seed(42)

lambda_values = simulate_ou_lambda(t_grid)

arrival_times = simulate_cox_process(lambda_values, t_grid)

plt.figure(figsize=(10,4))
plt.plot(t_grid, lambda_values, label='Stochastic $\lambda(t)$')
plt.vlines(arrival_times, ymin=0, ymax=np.max(lambda_values), color='r', alpha=0.4, label='Arrivals')
