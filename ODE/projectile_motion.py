def projectile_motion_with_drag(dt=0.01, total_time=5, cd=0.005, g=9.81, v0=100, angle=45):
    """
    Simulate 2D projectile motion with air resistance using the Euler method.
    
    Parameters:
    - dt: Time step in seconds.
    - total_time: Total simulation time in seconds.
    - cd: Drag coefficient.
    - g: Acceleration due to gravity in m/s^2.
    - v0: Initial velocity in m/s.
    - angle: Launch angle in degrees.
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Initial conditions
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    x, y = 0, 0
    
    # Time array
    t = np.arange(0, total_time, dt)
    
    # Initialize arrays to store positions
    x_vals, y_vals = [], []
    
    for _ in t:
        # Update velocities
        v = np.sqrt(vx**2 + vy**2)  # Total velocity
        vx -= cd * vx * v * dt
        vy -= (g + cd * vy * v) * dt
        
        # Update positions
        x += vx * dt
        y += vy * dt
        
        # Store positions
        x_vals.append(x)
        y_vals.append(y)
        
        # Stop if projectile hits the ground
        if y < 0:
            break
    
    return np.array(x_vals), np.array(y_vals)

# Parameters
dt = 0.01
total_time = 8
cd = 0.1  # Increased drag coefficient for more noticeable effect
v0 = 50  # Initial velocity in m/s
angle = 30  # Launch angle in degrees

# Simulate projectile motion
x_vals, y_vals = projectile_motion_with_drag(dt, total_time, cd, v0=v0, angle=angle)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='Trajectory with Air Resistance')
plt.title('Projectile Motion with Air Resistance')
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.legend()
plt.grid(True)
plt.show()
