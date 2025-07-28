import numpy as np
from boid import Boid

class BoidSimulation:
    def __init__(self, num_boids=50, config=None):
        self.config = config or {
            'v_max': 2.0,
            'neighborhood_radius': 1.5,
            'alignment_strength': 0.05,
            'cohesion_strength': 0.01,
            'separation_strength': 0.1,
            'bounds': np.array([10.0, 10.0]),
        }
        self.dt = 0.1
        self.boids = [
            Boid(
                position=np.random.rand(2) * self.config['bounds'],
                velocity=(np.random.rand(2) - 0.5) * 2,
                config=self.config
            )
            for _ in range(num_boids)
        ]

    def step(self):
        for boid in self.boids:
            neighbors = self.get_neighbors(boid)
            boid.compute_steering(neighbors)

        for boid in self.boids:
            boid.update_position(self.dt)

    def get_neighbors(self, boid):
        return [
            other for other in self.boids
            if other is not boid and
            np.linalg.norm(other.position - boid.position) < self.config['neighborhood_radius']
        ]

    def get_positions_velocities(self):
        pos = np.array([b.position for b in self.boids])
        vel = np.array([b.velocity for b in self.boids])
        return pos, vel
