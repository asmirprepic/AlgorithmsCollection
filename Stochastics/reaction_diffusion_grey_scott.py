import numpy as np
import matplotlib.pyplot as plt

size = 300
U = np.ones((size, size))
V = np.zeros((size, size))

Du_base, Dv_base = 0.16, 0.08
F_base, k_base = 0.045, 0.06
dt = 1.0
steps = 8000

def seed_droplets(n=10, radius=6):
    for _ in range(n):
        cx, cy = np.random.randint(radius, size-radius, size=2)
        y, x = np.ogrid[-radius:radius, -radius:radius]
        mask = x**2 + y**2 <= radius**2
        U[cx-radius:cx+radius, cy-radius:cy+radius][mask] = 0.50
        V[cx-radius:cx+radius, cy-radius:cy+radius][mask] = 0.25

seed_droplets(n=20, radius=6)

def laplace(Z):
    return (
        -4 * Z +
        np.roll(Z, 1, axis=0) +
        np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) +
        np.roll(Z, -1, axis=1)
    )

Fx = F_base + 0.01 * np.sin(np.linspace(0, 3 * np.pi, size))[:, None]
Fy = k_base + 0.01 * np.cos(np.linspace(0, 2 * np.pi, size))[None, :]
F = Fx + Fy  # F varies in a wave-like way across space
k = Fy + Fx.T

for i in range(steps):
    Lu = laplace(U)
    Lv = laplace(V)
    reaction = U * V**2

    Du = Du_base + 0.005 * np.sin(i / 300)
    Dv = Dv_base + 0.003 * np.cos(i / 500)

    U += (Du * Lu - reaction + F * (1 - U)) * dt
    V += (Dv * Lv + reaction - (F + k) * V) * dt

    if i % 1000 == 0:
        print(f"Step {i}")


plt.figure(figsize=(8, 8))
plt.imshow(V, cmap='magma', interpolation='bilinear')
plt.title("Gray-Scott Variant — Twisted Feed/Kill, Droplets")
plt.axis('off')
plt.tight_layout()
plt.show()
