
import numpy as np

class Boid:
    def __init__(self, position, velocity, config):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.config = config

    def limit_velocity(self):
        speed = np.linalg.norm(self.velocity)
        if speed > self.config['v_max']:
            self.velocity = self.velocity / speed * self.config['v_max']

    def update_position(self, dt):
        self.position += self.velocity * dt
        self.position %= self.config['bounds']  # wrap around

    def compute_steering(self, neighbors):
        if not neighbors:
            return
