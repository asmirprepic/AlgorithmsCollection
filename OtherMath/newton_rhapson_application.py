import numpy as np

def colebrook(Re, epsilon, D):
    # Iteratively solve the Colebrook equation for the friction factor f
    f = 0.02  # Initial guess
    for _ in range(10):
        f = 1.0 / (-2.0 * np.log10((epsilon / D) / 3.7 + 2.51 / (Re * np.sqrt(f))))
    return f

def newton_raphson_flow_rate(Delta_P, L, D, rho, mu, epsilon, initial_guess=1.0, tolerance=1e-6, max_iterations=100):
    v = initial_guess
    for i in range(max_iterations):
        Re = rho * v * D / mu
        f = colebrook(Re, epsilon, D)
        
        f_v = Delta_P - f * (L / D) * (rho / 2) * v**2
        f_prime_v = -2 * f * (L / D) * (rho / 2) * v
        
        v_next = v - f_v / f_prime_v
        
        if abs(v_next - v) < tolerance:
            print(f"Converged after {i+1} iterations.")
            return v_next
        
        v = v_next
        
    raise ValueError("Newton-Raphson method did not converge.")

# Given parameters
Delta_P = 5000  # Pressure drop in Pascals
L = 50  # Length of the pipe in meters
D = 0.3  # Diameter of the pipe in meters
rho = 1000  # Density of the fluid in kg/m³
mu = 0.001  # Dynamic viscosity of the fluid in Pa·s
epsilon = 0.0001  # Roughness height in meters

flow_velocity = newton_raphson_flow_rate(Delta_P, L, D, rho, mu, epsilon)
print(f"The flow velocity for the given pressure drop is approximately {flow_velocity:.6f} m/s.")
