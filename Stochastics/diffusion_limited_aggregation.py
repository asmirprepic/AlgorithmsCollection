import numpy as np
import matplotlib.pyplot as plt


GRID_SIZE = 201
CENTER = GRID_SIZE//2

NUM_P = 1000
MAX_STEPS = 500
STICK_RADIUS = 1

grid = np.zeros((GRID_SIZE,GRID_SIZE),dtype=int)
grid[CENTER,CENTER] = 1


def is_sticking(x,y):
    '''Check if cluster is approached'''
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            if dx == dy == 0:
                continue
            nx,ny = x + dx, y+dy

            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if grid[nx,ny] == 1:
                    return True
    return False

def random_walk_particle():
    '''For one poarticale'''
    angle = np.random.uniform(0,2*np.pi)
    r = CENTER * 0.9
    x = int(CENTER + r*np.cos(angle))
    y = int(CENTER + r *np.sin(angle))

    for _ in range(MAX_STEPS):
       dx,dy = np.random.choice([-1,0,1]),np.random.choice([-1,0,1])
       x += dx
       y += dy

       #check
       if x <= 1 or x >= GRID_SIZE -2 or y <= 1 or y >= GRID_SIZE -2:
           return # Got lost

       if is_sticking(x,y):
           grid[x,y] = 1
           return

# Simulation
for _ in range(NUM_P):
    random_walk_particle()

plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap='inferno', origin='lower')
plt.title(f"Diffusion-Limited Aggregation ({NUM_P} particles)")
plt.axis('off')
plt.tight_layout()
plt.show()

