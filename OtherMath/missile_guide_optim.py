import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from OtherMath.gradient_descent_class import GradientDescent



def missile_model(state, t, N, target_velocity):
    x_m, v_xm, y_m, v_ym = state
    x_t = target_velocity[0] * t
    y_t = target_velocity[1] * t
    
    # Line-of-sight angle
    theta_LOS = np.arctan2(y_t - y_m, x_t - x_m)
    
    # Missile speed
    v_m = np.sqrt(v_xm**2 + v_ym**2)
    
    # Accelerations
    a_x = N * v_m * np.sin(theta_LOS)
    a_y = N * v_m * np.cos(theta_LOS)
    
    dxmdt = v_xm
    dv_xmdt = a_x
    dymdt = v_ym
    dv_ymdt = a_y
    return [dxmdt, dv_xmdt, dymdt, dv_ymdt]

def model(params, t, initial_state, target_velocity):
    N = params[0]
    solution = odeint(missile_model, initial_state, t, args=(N, target_velocity))
    return solution

def cost_function(params, t, initial_state, x_m_observed, y_m_observed, target_velocity):
    solution = model(params, t, initial_state, target_velocity)
    x_m_predicted, _, y_m_predicted, _ = solution.T
    cost = np.mean((x_m_predicted - x_m_observed)**2 + (y_m_predicted - y_m_observed)**2)
    return cost


# Generate synthetic data
true_N = 3.0
target_velocity = [0.1, 0.1]  # Target moving at constant velocity
initial_state = [0, 1, 0, 1]
t = np.linspace(0, 10, 100)

solution = odeint(missile_model, initial_state, t, args=(true_N, target_velocity))
x_m, v_xm, y_m, v_ym = solution.T

# Add noise to the data
noise_level = 0.05
x_m_noisy = x_m + np.random.normal(0, noise_level, x_m.shape)
y_m_noisy = y_m + np.random.normal(0, noise_level, y_m.shape)

# Initial guesses for the parameters
initial_params = [2.0]

# Create GradientDescent instance
gd = GradientDescent(cost_function, learning_rate=0.01, max_iter=5000, tolerance=1e-6)

# Run optimization
estimated_params, history = gd.optimize(initial_params, t, initial_state, x_m_noisy, y_m_noisy, target_velocity)

# Results
print(f"Estimated parameters: N = {estimated_params[0]}")

# Plot the results
solution_estimated = model(estimated_params, t, initial_state, target_velocity)
x_m_estimated, _, y_m_estimated, _ = solution_estimated.T

plt.figure(figsize=(10, 6))
plt.plot(x_m_noisy, y_m_noisy, 'o', label='Noisy missile trajectory')
plt.plot(x_m, y_m, '-', label='True missile trajectory')
plt.plot(x_m_estimated, y_m_estimated, '--', label='Estimated missile trajectory')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.show()
