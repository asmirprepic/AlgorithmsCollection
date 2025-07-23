import numpy as np
import matplotlib.pyplot as plt

def lambda_t(t):
    # t in hours since 8 A.M.
    if 0 <= t < 3:
        return 5 + 5 * t
    elif 3 <= t < 5:
        return 20
    elif 5 <= t <= 9:
        return 20 - 2 * (t - 5)
    else:
        return 0

def simulate_nhpp(lambda_max, T_start, T_end):
    arrivals = []
    t = T_start
    while t < T_end:
        u = np.random.exponential(scale=1/lambda_max)
        t += u
        if t >= T_end:
            break
        acceptance_prob = lambda_t(t) / lambda_max
        if np.random.rand() < acceptance_prob:
            arrivals.append(t)
    return arrivals

np.random.seed(42)
arrivals = simulate_nhpp(lambda_max=20, T_start=0, T_end=9)


arrival_times = [8 + t for t in arrivals]  # in hours

plt.hist(arrival_times, bins=30, edgecolor='k')

plt.show()
