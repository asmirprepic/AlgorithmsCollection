# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euler_stochastic(deterministic_rate, stochastic_rate, initial_condition, parameters, deltaT, n_steps, D):
    # Initialize variables
    t = np.linspace(0, deltaT * n_steps, n_steps)
    x = np.zeros(n_steps)
    x[0] = initial_condition['x']
    
    r = parameters['r']
    K = parameters['K']
    
    for i in range(1, n_steps):
        deterministic_change = deterministic_rate(x[i-1], r, K)
        stochastic_change = stochastic_rate(x[i-1], D, deltaT)
        x[i] = x[i-1] + deterministic_change * deltaT + stochastic_change * np.sqrt(deltaT)
        # Ensure population does not go negative
        x[i] = max(x[i], 0)
        
    return pd.DataFrame({'t': t, 'x': x})

def deterministic_logistic(x, r, K):
    return r * x * (1 - x / K)

def stochastic_logistic(x, D, deltaT):
    return D * x * np.random.normal(0, 1)

def main():
    # Initial condition and parameters
    init_logistic = {'x': 3}
    logistic_parameters = {'r': 0.8, 'K': 100}
    deltaT_logistic = 0.05
    timesteps_logistic = 200
    D_logistic = 1

    # Simulate the stochastic logistic equation
    logistic_out = euler_stochastic(
        deterministic_rate=deterministic_logistic,
        stochastic_rate=stochastic_logistic,
        initial_condition=init_logistic,
        parameters=logistic_parameters,
        deltaT=deltaT_logistic,
        n_steps=timesteps_logistic,
        D=D_logistic
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(logistic_out['t'], logistic_out['x'], color='black')
    plt.axhline(logistic_parameters['K'], color='red', linestyle='--', label='Carrying Capacity $K$')
    plt.xlabel('Time $t$')
    plt.ylabel('Population $x$')
    plt.title('Stochastic Logistic Equation')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
