import numpy as np
import pygame

class Lidar2D:
    def __init__(self, range_max=20, angle_range=2*np.pi, 
                 resolution=1.0, noise_std=0.1, meter_to_pixel=50.0):  # resolution in degrees
        """
        Initialize 2D LiDAR sensor.
        
        Args:
            range_max (float): Maximum detection range in meters
            angle_range (float): Angular range of the sensor in radians
            resolution (float): Angular resolution in degrees
            noise_std (float): Standard deviation of measurement noise
        """
        self.range_max = range_max
        self.angle_range = angle_range
        self.resolution = np.deg2rad(resolution)  # Convert to radians
        self.noise_std = noise_std
        self.meter_to_pixel = meter_to_pixel
        
        # Calculate number of beams based on resolution
        self.num_beams = int(self.angle_range / self.resolution)
        self.angles = np.linspace(0, self.angle_range, self.num_beams, endpoint=False)
        self.point_cloud = []  # Store point cloud data

    def get_readings(self, robot_pos, robot_orientation, obstacles):
        """
        Get LiDAR readings using geometric ray-obstacle intersection.
        
        Args:
            robot_pos: Robot position in pixels
            robot_orientation: Robot orientation in radians
            obstacles: List of pygame.Rect obstacles
        """
        readings = np.full(self.num_beams, self.range_max)
        robot_pos = robot_pos / self.meter_to_pixel  # Convert to meters
        self.point_cloud = []  # Clear previous point cloud
        
        for i, angle in enumerate(self.angles):
            # Calculate global beam angle
            global_angle = robot_orientation + angle
            
            # Calculate beam direction vector
            beam_dir = np.array([
                np.cos(global_angle),
                np.sin(global_angle)
            ])
            
            # Track closest intersection for this ray
            closest_distance = self.range_max
            closest_intersection = None
            
            # Check intersection with each obstacle
            for obstacle in obstacles:
                # Convert obstacle coordinates to meters
                obstacle_left = obstacle.left / self.meter_to_pixel
                obstacle_right = obstacle.right / self.meter_to_pixel
                obstacle_top = obstacle.top / self.meter_to_pixel
                obstacle_bottom = obstacle.bottom / self.meter_to_pixel
                
                # Create line segments for obstacle edges
                segments = [
                    # Top line
                    (np.array([obstacle_left, obstacle_top]),
                     np.array([obstacle_right, obstacle_top])),
                    # Right line
                    (np.array([obstacle_right, obstacle_top]),
                     np.array([obstacle_right, obstacle_bottom])),
                    # Bottom line
                    (np.array([obstacle_right, obstacle_bottom]),
                     np.array([obstacle_left, obstacle_bottom])),
                    # Left line
                    (np.array([obstacle_left, obstacle_bottom]),
                     np.array([obstacle_left, obstacle_top]))
                ]
                
                # Check intersection with each segment
                for start, end in segments:
                    intersection = self._ray_segment_intersection(
                        robot_pos, beam_dir, start, end)
                    if intersection is not None:
                        distance = np.linalg.norm(intersection - robot_pos)
                        # Only keep the closest intersection for this ray
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_intersection = intersection
            
            # If we found an intersection for this ray, record it
            if closest_intersection is not None:
                readings[i] = closest_distance
                # Store point cloud data in meters
                self.point_cloud.append({
                    'x': closest_intersection[0],
                    'y': closest_intersection[1],
                    'distance': closest_distance,
                    'angle': global_angle
                })

        # Add noise
        readings += np.random.normal(0, self.noise_std, self.num_beams)
        readings = np.clip(readings, 0, self.range_max)
        
        return readings

    def _ray_segment_intersection(self, ray_origin, ray_dir, segment_start, segment_end):
        """
        Calculate intersection between a ray and a line segment using vector algebra.
        Returns intersection point if it exists, None otherwise.
        """
        # Calculate vectors
        v1 = ray_origin - segment_start
        v2 = segment_end - segment_start
        v3 = np.array([-ray_dir[1], ray_dir[0]])

        # Calculate dot product
        dot = np.dot(v2, v3)
        if abs(dot) < 1e-8:  # Nearly parallel
            return None

        # Calculate intersection parameters
        t1 = np.cross(v2, v1) / dot
        t2 = np.dot(v1, v3) / dot

        # Check if intersection exists
        if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
            return ray_origin + t1 * ray_dir
        return None 

    def visualize_rays(self, robot_pos, robot_orientation, obstacles, output_filename=None):
        """
        Generate a visualization of LiDAR rays to help debug occlusion issues.
        
        Args:
            robot_pos: Robot position in pixels
            robot_orientation: Robot orientation in radians
            obstacles: List of pygame.Rect obstacles
            output_filename: Optional filename to save the visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Convert robot position to meters
        robot_pos_meters = robot_pos / self.meter_to_pixel
        
        # Plot the robot
        robot_circle = plt.Circle(
            (robot_pos_meters[0], robot_pos_meters[1]),
            0.3,  # Arbitrary size for visibility
            color='green',
            alpha=0.7
        )
        plt.gca().add_patch(robot_circle)
        
        # Plot the obstacles
        for obstacle in obstacles:
            # Convert obstacle coordinates to meters
            left = obstacle.left / self.meter_to_pixel
            right = obstacle.right / self.meter_to_pixel
            top = obstacle.top / self.meter_to_pixel
            bottom = obstacle.bottom / self.meter_to_pixel
            
            # Create and add rectangle
            rect = patches.Rectangle(
                (left, top),
                right - left,
                bottom - top,
                linewidth=2,
                edgecolor='blue',
                facecolor='blue',
                alpha=0.3
            )
            plt.gca().add_patch(rect)
        
        # Generate and plot rays
        for angle in self.angles:
            # Calculate global angle
            global_angle = robot_orientation + angle
            
            # Calculate beam direction vector
            beam_dir = np.array([
                np.cos(global_angle),
                np.sin(global_angle)
            ])
            
            # Default ray endpoint (at maximum range)
            ray_end = robot_pos_meters + beam_dir * self.range_max
            
            # Track closest intersection for this ray
            closest_distance = self.range_max
            closest_intersection = None
            
            # Check intersection with each obstacle
            for obstacle in obstacles:
                # Convert obstacle coordinates to meters
                obstacle_left = obstacle.left / self.meter_to_pixel
                obstacle_right = obstacle.right / self.meter_to_pixel
                obstacle_top = obstacle.top / self.meter_to_pixel
                obstacle_bottom = obstacle.bottom / self.meter_to_pixel
                
                # Create line segments for obstacle edges
                segments = [
                    # Top line
                    (np.array([obstacle_left, obstacle_top]),
                     np.array([obstacle_right, obstacle_top])),
                    # Right line
                    (np.array([obstacle_right, obstacle_top]),
                     np.array([obstacle_right, obstacle_bottom])),
                    # Bottom line
                    (np.array([obstacle_right, obstacle_bottom]),
                     np.array([obstacle_left, obstacle_bottom])),
                    # Left line
                    (np.array([obstacle_left, obstacle_bottom]),
                     np.array([obstacle_left, obstacle_top]))
                ]
                
                # Check intersection with each segment
                for start, end in segments:
                    intersection = self._ray_segment_intersection(
                        robot_pos_meters, beam_dir, start, end)
                    if intersection is not None:
                        distance = np.linalg.norm(intersection - robot_pos_meters)
                        # Only keep the closest intersection for this ray
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_intersection = intersection
            
            # Update ray endpoint if we found an intersection
            if closest_intersection is not None:
                ray_end = closest_intersection
            
            # Draw the ray
            plt.plot(
                [robot_pos_meters[0], ray_end[0]],
                [robot_pos_meters[1], ray_end[1]],
                'r-', linewidth=0.5, alpha=0.3
            )
            
            # Mark intersection point if it exists
            if closest_intersection is not None:
                plt.plot(
                    closest_intersection[0],
                    closest_intersection[1],
                    'ro', markersize=3
                )
        
        # Set up the plot
        plt.xlabel('X Position (meters)')
        plt.ylabel('Y Position (meters)')
        plt.title('LiDAR Ray Visualization')
        plt.grid(True)
        plt.axis('equal')
        
        # Save if requested
        if output_filename:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Ray visualization saved as '{output_filename}'")
        
        # Display
        plt.show()

    def get_point_cloud(self):
        """Return the current point cloud data"""
        return self.point_cloud 