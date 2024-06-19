import numpy as np
import matplotlib.pyplot as plt

def numerical_gradient(func, x, h=1e-6):
    """Compute the numerical gradient of func at x using central difference."""
    grad = (func(x + h) - func(x - h)) / (2 * h)
    return grad

def gradient_ascent(func, initial_x, learning_rate=0.01, tol=1e-6, max_iter=1000):
    x = initial_x
    x_values = [x]
    for _ in range(max_iter):
        gradient = numerical_gradient(func, x)
        new_x = x + learning_rate * gradient
        
        # Check for convergence
        if np.abs(new_x - x) < tol:
            break
        
        x = new_x
        x_values.append(x)
    return x, x_values

def maximize_area_in_ellipse():
    def area(x):
        y = np.sqrt(1 - (x**2 / 4))  # derived from the ellipse equation
        return 4 * x * y

    # Initial guess for x
    initial_x = 1.0

    # Perform gradient ascent
    x_opt, x_values = gradient_ascent(area, initial_x)
    
    # Calculate optimal y and area
    y_opt = np.sqrt(1 - (x_opt**2 / 4))
    max_area = area(x_opt)
    
    return x_opt, y_opt, max_area, x_values

def plot_results():
    # Execute the function and get results
    x_opt, y_opt, max_area, x_values = maximize_area_in_ellipse()

    # Define the area function for plotting
    def area(x):
        y = np.sqrt(1 - (x**2 / 4))
        return 4 * x * y

    # Generate x values for plotting the area function
    x_range = np.linspace(-2, 2, 400)
    area_values = area(x_range)

    # Plot the area function
    plt.figure(figsize=(12, 6))
    plt.plot(x_range, area_values, label="Area function", color='blue')
    
    # Plot the progression of x values during gradient ascent
    plt.scatter(x_values, [area(x) for x in x_values], color='red', label="Gradient ascent steps", zorder=5)
    plt.plot(x_values, [area(x) for x in x_values], color='red', linestyle='--', alpha=0.6)
    
    # Highlight the optimal point
    plt.scatter([x_opt], [max_area], color='green', label="Optimal point", zorder=10)
    
    # Annotate the optimal point
    plt.annotate(f'Optimal x: {x_opt:.3f}\nOptimal y: {y_opt:.3f}\nMax area: {max_area:.3f}',
                 xy=(x_opt, max_area), xytext=(x_opt + 0.5, max_area - 5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10)
    
    # Plot settings
    plt.xlabel('x')
    plt.ylabel('Area')
    plt.title('Maximizing the Area of a Rectangle Inscribed in an Ellipse')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the plotting function
plot_results()
