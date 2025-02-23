import pygame
from simulator import Simulator

# Create an instance of the simulator
sim = Simulator(map_name="default", size=0.3, max_velocity=5, mass=8, use_lidar=True, lidar_resolution=5.0)

def control_logic():
    # Get keyboard input
    keys = pygame.key.get_pressed()
    
    # Initialize target velocities
    Vx_target = 0.0
    Vy_target = 0.0
    w_target = 0.0
    
    # Set velocities based on key presses
    if keys[pygame.K_LEFT]:
        Vx_target = -sim.max_velocity
    elif keys[pygame.K_RIGHT]:
        Vx_target = sim.max_velocity

    if keys[pygame.K_UP]:
        Vy_target = -sim.max_velocity
    elif keys[pygame.K_DOWN]:
        Vy_target = sim.max_velocity

    if keys[pygame.K_a]:
        w_target = -3.14  # Approximately -180 degrees
    elif keys[pygame.K_d]:
        w_target = 3.14   # Approximately 180 degrees

    return Vx_target, Vy_target, w_target

# Run the simulation
sim.run(control_logic)
