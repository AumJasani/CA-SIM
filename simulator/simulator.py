import pygame
import numpy as np
import matplotlib.pyplot as plt
from .config import *
from .physics import PhysicsEngine
from .maps import MapManager
from .rendering import Renderer
from .sensors import Lidar2D

class Simulator:
    def __init__(self, map_name="default", start_position=None, goal_position=None, 
                 max_velocity=DEFAULT_MAX_VELOCITY, mass=DEFAULT_MASS, size=DEFAULT_SIZE,
                 use_lidar=False, lidar_resolution=1.0, lidar_range_max=5.0):
        pygame.init()
        
        # Initialize components
        self.physics = PhysicsEngine(mass, DEFAULT_FRICTION_COEFF, DEFAULT_RESTITUTION)
        self.map_manager = MapManager(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        self.renderer = Renderer(DEFAULT_WIDTH, DEFAULT_HEIGHT, 
                               {"BLACK": BLACK, "RED": RED, "GREEN": GREEN, 
                                "BLUE": BLUE, "WHITE": WHITE})
        
        # LiDAR flag and initialization
        self.use_lidar = use_lidar
        self.lidar = None
        if self.use_lidar:
            self.lidar = Lidar2D(
                range_max=lidar_range_max,  # 5 meters
                angle_range=2*np.pi,  # 360 degrees
                resolution=lidar_resolution,  # 1 degree resolution
            )
        
        # Robot specifications
        self.max_velocity = max_velocity * METER_TO_PIXEL
        self.circle_radius = int(size * METER_TO_PIXEL)
        
        # Load map and set positions
        map_data = self.map_manager.load_map(map_name)
        self.obstacles = map_data["obstacles"]
        self.circle_pos = np.array(start_position if start_position else map_data["start"], dtype=float)
        self.goal_pos = np.array(goal_position if goal_position else map_data["goal"])
        self.goal_radius = int(DEFAULT_GOAL_RADIUS_METERS * METER_TO_PIXEL)
        
        # Data recording for plotting
        self.time_points = []
        self.target_velocities_x = []
        self.target_velocities_y = []
        self.target_velocities_w = []
        self.actual_velocities_x = []
        self.actual_velocities_y = []
        self.actual_velocities_w = []
        self.time = 0.0
        
        # Simulation clock
        self.clock = pygame.time.Clock()
        self.dt = 1 / DEFAULT_FPS

    def run(self, control_function):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get target velocities and update physics
            target_vx, target_vy, target_w = control_function()
            target_velocity = np.array([target_vx, target_vy]) * METER_TO_PIXEL
            velocity, angular_velocity, orientation = self.physics.update(
                target_velocity, target_w, self.dt)
            
            # Update position
            self.circle_pos += velocity * self.dt
            
            # Handle collisions
            for obstacle in self.obstacles:
                position_correction = self.physics.handle_collision(
                    self.circle_pos, self.circle_radius, obstacle)
                self.circle_pos += position_correction
            
            # Check if goal is reached
            distance_to_goal = np.linalg.norm(self.circle_pos - self.goal_pos)
            if distance_to_goal < (self.circle_radius + self.goal_radius):
                running = False
                break
            
            # Get LiDAR readings and render complete scene
            lidar_readings = None
            if self.use_lidar:
                lidar_readings = self.lidar.get_readings(
                    self.circle_pos,
                    orientation,
                    self.obstacles
                )
                lidar_readings *= METER_TO_PIXEL  # Convert to pixels for rendering
                # Log lidar readings to file
                # with open('lidar_readings.txt', 'a') as f:
                #     f.write(f"Time: {self.time:.2f}, Readings: {lidar_readings.tolist()}\n")
            
            # Render complete scene
            self.renderer.render_scene(
                self.circle_pos, 
                self.circle_radius,
                self.goal_pos, 
                self.goal_radius,
                self.obstacles, 
                orientation,
                self.use_lidar,
                lidar_readings
            )
            
            pygame.display.flip()
            self.clock.tick(DEFAULT_FPS)
            
            # Record data
            self.time_points.append(self.time)
            self.target_velocities_x.append(target_vx)
            self.target_velocities_y.append(target_vy)
            self.target_velocities_w.append(target_w)
            self.actual_velocities_x.append(velocity[0] / METER_TO_PIXEL)
            self.actual_velocities_y.append(velocity[1] / METER_TO_PIXEL)
            self.actual_velocities_w.append(angular_velocity)
            self.time += self.dt
        
        # After simulation ends, plot the velocities
        self.plot_velocities()
        pygame.quit()

    def plot_velocities(self):
        """Plot the recorded velocity data."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot X velocities
        ax1.plot(self.time_points, self.target_velocities_x, 'b--', label='Target Vx')
        ax1.plot(self.time_points, self.actual_velocities_x, 'b-', label='Actual Vx')
        ax1.set_ylabel('X Velocity (m/s)')
        ax1.set_xlabel('Time (s)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Y velocities
        ax2.plot(self.time_points, self.target_velocities_y, 'r--', label='Target Vy')
        ax2.plot(self.time_points, self.actual_velocities_y, 'r-', label='Actual Vy')
        ax2.set_ylabel('Y Velocity (m/s)')
        ax2.set_xlabel('Time (s)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot angular velocities
        ax3.plot(self.time_points, self.target_velocities_w, 'g--', label='Target ω')
        ax3.plot(self.time_points, self.actual_velocities_w, 'g-', label='Actual ω')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angular Velocity (rad/s)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
