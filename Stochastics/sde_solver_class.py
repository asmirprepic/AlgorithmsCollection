import numpy as np

class SDESolver:
    def __init__(self, drift, diffusion, x0, t0, T, dt):
        self.drift = drift
        self.diffusion = diffusion
        self.x0 = x0
        self.t0 = t0
        self.T = T
        self.dt = dt
        self.n_steps = int((T - t0) / dt)
    
    def euler_maruyama(self):
        t = np.linspace(self.t0, self.T, self.n_steps)
        x = np.zeros((self.n_steps, len(self.x0)))
        x[0] = self.x0
        
        for i in range(1, self.n_steps):
            dt = t[i] - t[i-1]
            dw = np.sqrt(dt) * np.random.normal(size=len(self.x0))
            x[i] = x[i-1] + self.drift(x[i-1], t[i-1]) * dt + self.diffusion(x[i-1], t[i-1]) * dw
        
        return t, x
    
    def milstein(self):
        t = np.linspace(self.t0, self.T, self.n_steps)
        x = np.zeros((self.n_steps, len(self.x0)))
        x[0] = self.x0
        
        for i in range(1, self.n_steps):
            dt = t[i] - t[i-1]
            dw = np.sqrt(dt) * np.random.normal(size=len(self.x0))
            x[i] = x[i-1] + self.drift(x[i-1], t[i-1]) * dt + self.diffusion(x[i-1], t[i-1]) * dw + 0.5 * self.diffusion(x[i-1], t[i-1]) * (dw**2 - dt)
        
        return t, x
