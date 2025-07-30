import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_simulation(simulation, steps=200, interval=50):
    fig, ax = plt.subplots()
    pos, vel = simulation.get_positions_velocities()
    scat = ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1])

    ax.set_xlim(0, simulation.config['bounds'][0])
    ax.set_ylim(0, simulation.config['bounds'][1])
    ax.set_title("Flocking Simulation (Boids)")

    def update(frame):
        simulation.step()
        pos, vel = simulation.get_positions_velocities()
        scat.set_offsets(pos)
        scat.set_UVC(vel[:, 0], vel[:, 1])
        return scat,

    ani = FuncAnimation(fig, update, frames=steps, interval=interval, blit=False)
    plt.show()
