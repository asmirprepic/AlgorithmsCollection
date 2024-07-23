import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 0.01  # fuel consumption coefficient for v^2
b = 0.1   # fuel consumption coefficient for v
c = 1     # constant fuel consumption rate
D = 1000  # total distance in km
k = 50    # penalty cost per hour

# Total cost function
def total_cost(v):
    return (a * v**2 + b * v + c) * D + (k * D) / v

# Derivative of the total cost function
def d_total_cost_dv(v):
    return 2 * a * v * D + b * D - (k * D) / v**2

def finite_difference_gradient(f,x,h = 1e-5):
    grad_x = (f(x+h) - f(x))/h
    return grad_x

# Newton-Raphson method
def newton_raphson(f, df, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = finite_difference_gradient(f,x)
        x_new = x - f(x) / grad
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError("Newton-Raphson method did not converge")

# Initial guess
v_0 = 50  # initial guess for speed in km/h

# Finding the optimal speed
v_opt = newton_raphson(total_cost, d_total_cost_dv, v_0)

# Calculating the minimum total cost
C_min = total_cost(v_opt)

print(f"Optimal speed: {v_opt} km/h")
print(f"Minimum total cost: {C_min} currency units")

# Plotting the results
v_values = np.linspace(10, 120, 1000)
C_values = total_cost(v_values)

plt.plot(v_values, C_values, label='Total Cost')
plt.axvline(x=v_opt, color='r', linestyle='--', label=f'Optimal v = {v_opt:.2f} km/h')
plt.xlabel('Speed (km/h)')
plt.ylabel('Total Cost')
plt.title('Total Cost vs. Speed')
