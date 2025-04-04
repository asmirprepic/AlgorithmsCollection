def monte_carlo_pi(num_samples: int):
    # Generate random points (x, y) within the unit square [0, 1] x [0, 1]
    x = np.random.uniform(0, 1, num_samples)
    y = np.random.uniform(0, 1, num_samples)

    # Calculate how many points fall inside the quarter-circle (radius=1)
    inside_circle = (x**2 + y**2) <= 1
    num_inside_circle = np.sum(inside_circle)

    # Pi estimate: ratio of inside points * 4
    pi_estimate = 4 * num_inside_circle / num_samples

    return pi_estimate, x, y, inside_circle
