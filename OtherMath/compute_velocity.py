def compute_velocity(psi, dx, dy):
    """
    Compute velocity components from the streamfunction.
    """
    u = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dy)
    v = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dx)
    return u, v
