def solve_fokker_planck(
    mu: float,
    sigma: float,
    x_min: float,
    x_max: float,
    Nx: int,
    T: float,
    Nt: int,
    S0: float
):
   
    dx = (x_max - x_min) / Nx
    dt = T / Nt
    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, T, Nt)

   
    p0 = np.exp(-0.5 * ((x - S0) / (0.01 * S0))**2)
    p0 /= simps(p0, x)

    P = np.zeros((Nt, Nx))
    P[0] = p0

   
    def drift(x): return mu * x
    def diffusion(x): return 0.5 * sigma**2 * x**2

    for n in range(0, Nt - 1):
        p = P[n]
      
        dpdx = np.zeros(Nx)
        d2pdx2 = np.zeros(Nx)

        dpdx[1:-1] = (drift(x[2:]) * p[2:] - drift(x[:-2]) * p[:-2]) / (2 * dx)
        d2pdx2[1:-1] = (diffusion(x[2:]) * p[2:] - 2 * diffusion(x[1:-1]) * p[1:-1] + diffusion(x[:-2]) * p[:-2]) / dx**2

        P[n + 1] = P[n] + dt * (-dpdx + d2pdx2)

        P[n + 1] /= simps(P[n + 1], x)

    return x, t, P
