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
                 max_velocity=DEFAULT_MAX_VELOCITY, max_acceleration=DEFAULT_MAX_ACCELERATION, mass=DEFAULT_MASS, size=0.3,
                 use_lidar=False, lidar_resolution=1.0, lidar_range_max=5.0,
                 plot_velocities_graph=False, plot_energies_graph=False, plot_lidar_map_graph=False):
        pygame.init()
        
        # Initialize components
        self.map_manager = MapManager()
        map_data = self.map_manager.load_map(map_name)
        self.window_width = map_data["width"]
        self.window_height = map_data["height"]
        self.METER_TO_PIXEL = int(map_data["conv"])
        self.physics = PhysicsEngine(mass, DEFAULT_FRICTION_COEFF, DEFAULT_RESTITUTION, size, max_acceleration, self.METER_TO_PIXEL)
        self.renderer = Renderer(self.window_width, self.window_height, 
                               {"BLACK": BLACK, "RED": RED, "GREEN": GREEN, 
                                "BLUE": BLUE, "WHITE": WHITE})
        
        # LiDAR flag and initialization
        self.use_lidar = use_lidar
        self.lidar = None
        self.latest_point_cloud = None
        # Store all point clouds for plotting
        self.all_point_clouds = []
        # Point cloud sampling settings for memory efficiency
        self.point_cloud_sample_rate = 5  # Store every Nth point cloud
        self.point_cloud_counter = 0
        
        if self.use_lidar:
            self.lidar = Lidar2D(
                range_max=lidar_range_max,  # 5 meters
                angle_range=2*np.pi,  # 360 degrees
                resolution=lidar_resolution,  # 1 degree resolution
                meter_to_pixel=self.METER_TO_PIXEL
            )
        
        # Robot specifications
        self.max_velocity = max_velocity
        self.circle_radius = float(size * self.METER_TO_PIXEL)
        print(f"Circle radius: {self.circle_radius}")
        
        # Load map and set positions
        self.obstacles = map_data["obstacles"]
        self.start_pos = np.array(start_position if start_position else map_data["start"], dtype=float)
        self.goal_pos = np.array(goal_position if goal_position else map_data["goal"], dtype=float)
        self.goal_radius = int(DEFAULT_GOAL_RADIUS_METERS * self.METER_TO_PIXEL)
        self.circle_pos = self.start_pos
        
        # Data recording for plotting
        self.time_points = []
        self.target_velocities_x = []
        self.target_velocities_y = []
        self.target_velocities_w = []
        self.actual_velocities_x = []
        self.actual_velocities_y = []
        self.actual_velocities_w = []
        self.time = 0.0
        
        # Collision flag
        self.collision_occurred = False

        # Plotting flags
        self.plot_velocities_graph = plot_velocities_graph
        self.plot_energies_graph = plot_energies_graph
        self.plot_lidar_map_graph = plot_lidar_map_graph

        # Simulation clock
        self.clock = pygame.time.Clock()
        self.dt = 1 / DEFAULT_FPS

    def get_latest_point_cloud(self):
        """Return the latest LiDAR point cloud data"""
        if not self.use_lidar:
            return None
        return self.latest_point_cloud

    def run(self, control_function):
        # Initialize energy history
        self.energy_history = {
            'time': [],
            'translational': [],
            'rotational': [],
            'elastic': [],
            'dissipated': [],
            'total': []
        }
        
        # Initialize target velocities
        target_vx, target_vy, target_w = 0.0,0.0,0.0

        # Track robot position history
        self.position_history = []
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Reset collision flag
            self.collision_occurred = False

            target_velocity = np.array([target_vx, target_vy]) * self.METER_TO_PIXEL
            velocity, angular_velocity, orientation = self.physics.update(
                target_velocity, target_w, self.dt)
            
            self.velocity = velocity
            self.angular_velocity = angular_velocity
            self.orientation = orientation
            # Update position
            self.circle_pos += velocity * self.dt
            
            # Handle collisions
            for obstacle in self.obstacles:
                position_correction, collision_occurred = self.physics.handle_collision(
                    self.circle_pos, self.circle_radius, obstacle)
                self.circle_pos += position_correction
                self.collision_occurred = self.collision_occurred or collision_occurred
            
            # Check if goal is reached
            distance_to_goal = np.linalg.norm(self.circle_pos - self.goal_pos)
            if distance_to_goal < (self.circle_radius + self.goal_radius):
                running = False
                break
            
            # Get LiDAR readings and point cloud
            lidar_readings = None
            if self.use_lidar:
                lidar_readings = self.lidar.get_readings(
                    self.circle_pos,
                    orientation,
                    self.obstacles
                )
                lidar_readings *= self.METER_TO_PIXEL  # Convert to pixels for rendering
                self.latest_point_cloud = self.lidar.get_point_cloud()  # Store latest point cloud
                
                # Store point cloud periodically to prevent memory issues
                self.point_cloud_counter += 1
                if self.point_cloud_counter >= self.point_cloud_sample_rate:
                    self.all_point_clouds.append(self.latest_point_cloud)  # Store point cloud
                    self.point_cloud_counter = 0
            
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
            self.actual_velocities_x.append(velocity[0] / self.METER_TO_PIXEL)
            self.actual_velocities_y.append(velocity[1] / self.METER_TO_PIXEL)
            self.actual_velocities_w.append(angular_velocity)
            self.time += self.dt
            
            # Record robot position for path tracking
            self.position_history.append(np.copy(self.circle_pos))
            
            # Track energies
            energies = self.physics.calculate_energies()
            self.energy_history['time'].append(self.time)
            self.energy_history['translational'].append(energies['translational'])
            self.energy_history['rotational'].append(energies['rotational'])
            self.energy_history['elastic'].append(energies['elastic'])
            self.energy_history['dissipated'].append(energies['dissipated'])
            total_energy = (energies['total_kinetic'] + 
                          energies['elastic'] - 
                          energies['dissipated'])
            self.energy_history['total'].append(total_energy)

            # Get target velocities from control function
            target_vx, target_vy, target_w = control_function()
        
        # After simulation ends, plot the velocities
        if self.plot_velocities_graph:
            self.plot_velocities()
        if self.plot_energies_graph:
            self.plot_energies()
        if self.plot_lidar_map_graph:
            self.plot_lidar_obstacle_map()
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

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'velocity_plot_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_energies(self):
        # Calculate cumulative energy consumption over time
        time_points = self.energy_history['time']
        translational_cumulative = np.cumsum(self.energy_history['translational'])
        total_cumulative = np.cumsum(self.energy_history['total'])
        dissipated_cumulative = np.cumsum(self.energy_history['dissipated'])

        # Create single plot
        plt.figure(figsize=(10, 6))
        
        # Plot cumulative energy consumption for each type
        plt.plot(time_points, translational_cumulative, '--', label='Translational KE')
        plt.plot(time_points, total_cumulative, '-', label='Total Energy')
        plt.plot(time_points, dissipated_cumulative, '--', label='Dissipated Energy')

        # Mark collision points
        if hasattr(self, 'collision_times'):
            for collision_time in self.collision_times:
                plt.axvline(x=collision_time, color='r', linestyle=':', alpha=0.5)
                plt.text(collision_time, plt.ylim()[1], 'Collision', rotation=90, 
                        verticalalignment='top')

        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Energy (J)')
        plt.title('Total Energy Consumption Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'energy_consumption_plot_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()


    def plot_lidar_obstacle_map(self):
        """Plot all obstacles detected by the LiDAR sensor throughout the simulation and save as PNG."""
        if not self.use_lidar or not self.all_point_clouds:
            print("No LiDAR data available to plot")
            return
            
        # Create a new figure
        plt.figure(figsize=(12, 10))
        
        # Extract x and y coordinates from all point clouds
        x_coords = []
        y_coords = []
        
        # Process all collected point clouds
        for point_cloud in self.all_point_clouds:
            for point in point_cloud:
                try:
                    # Handle different point formats
                    if isinstance(point, dict) and 'x' in point and 'y' in point:
                        x_coords.append(point['x'])
                        y_coords.append(point['y'])
                    elif isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                        x_coords.append(point[0])
                        y_coords.append(point[1])
                except Exception as e:
                    continue
        
        # Plot the obstacle points from LiDAR
        plt.scatter(x_coords, y_coords, s=2, color='red', alpha=0.5, label='LiDAR Points')
 
        
        # Plot the robot's path
        # Convert pixels to meters
        robot_path_x = [pos[0]/self.METER_TO_PIXEL for pos in self.position_history] if hasattr(self, 'position_history') else []
        robot_path_y = [pos[1]/self.METER_TO_PIXEL for pos in self.position_history] if hasattr(self, 'position_history') else []
        
        # If we have position history, plot the robot's path
        if robot_path_x and robot_path_y:
            plt.plot(robot_path_x, robot_path_y, 'g-', linewidth=2, label='Robot Path')
            
            # Mark start and end positions
            plt.plot(robot_path_x[0], robot_path_y[0], 'go', markersize=10, label='Start')
            plt.plot(robot_path_x[-1], robot_path_y[-1], 'mo', markersize=10, label='End')
            
            # Plot the robot's final position (circle with radius)
            robot_radius_meters = self.circle_radius / 50.0
            circle = plt.Circle(
                (robot_path_x[-1], robot_path_y[-1]), 
                robot_radius_meters, 
                fill=True, 
                color='green', 
                alpha=0.3
            )
            plt.gca().add_patch(circle)
        
        # Add plot details
        plt.xlabel('X Position (meters)')
        plt.ylabel('Y Position (meters)')
        plt.title('LiDAR Obstacle Map with Robot Path')
        plt.grid(True)
        plt.axis('equal')
        plt.legend(loc='upper right')
        
        
        # Calculate and display some statistics
        if x_coords and y_coords:
            total_points = len(x_coords)
            unique_points = len(set(zip(np.round(x_coords, 2), np.round(y_coords, 2))))
            plt.figtext(0.02, 0.02, f'Total points: {total_points}, Unique points: {unique_points}', 
                       fontsize=9, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        # Save the figure with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'lidar_obstacle_map_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"LiDAR obstacle map saved as '{filename}'")
        
        # Also show the plot
        plt.show()

    def visualize_lidar_rays(self, output_filename=None):
        """
        Visualize LiDAR rays for debugging the occlusion behavior.
        
        Args:
            output_filename: Optional filename to save the visualization
        """
        if not hasattr(self, 'lidar'):
            print("No LiDAR sensor available for visualization")
            return
            
        # Call the LiDAR's visualization method
        self.lidar.visualize_rays(
            robot_pos=self.circle_pos, 
            robot_orientation=self.physics.orientation,
            obstacles=self.obstacles,
            output_filename=output_filename
        )
