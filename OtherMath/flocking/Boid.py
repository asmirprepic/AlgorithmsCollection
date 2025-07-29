
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
        self.position %= self.config['bounds']  #

    def compute_steering(self, neighbors):
        if not neighbors:
            return

        avg_vel = np.mean([b.velocity for b in neighbors], axis=0)
        alignment = (avg_vel - self.velocity) * self.config['alignment_strength']

        center_mass = np.mean([b.position for b in neighbors], axis=0)
        cohesion = (center_mass - self.position) * self.config['cohesion_strength']


        separation = np.zeros(2)
        for other in neighbors:
            diff = self.position - other.position
            dist = np.linalg.norm(diff)
            if dist > 0:
                separation += diff / dist**2
        separation *= self.config['separation_strength']

        self.velocity += alignment + cohesion + separation
        self.limit_velocity()
