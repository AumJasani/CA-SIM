import numpy as np
import pygame

class Lidar2D:
    def __init__(self, range_max=20, angle_range=2*np.pi, 
                 resolution=1.0):  # resolution in degrees
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
        self.noise_std = 0.1
        
        # Calculate number of beams based on resolution
        self.num_beams = int(self.angle_range / self.resolution)
        self.angles = np.linspace(0, self.angle_range, self.num_beams, endpoint=False)

    def get_readings(self, robot_pos, robot_orientation, obstacles):
        """
        Get LiDAR readings using geometric ray-obstacle intersection.
        
        Args:
            robot_pos: Robot position in pixels
            robot_orientation: Robot orientation in radians
            obstacles: List of pygame.Rect obstacles
        """
        readings = np.full(self.num_beams, self.range_max)
        robot_pos = robot_pos / 50.0  # Convert to meters
        
        for i, angle in enumerate(self.angles):
            # Calculate global beam angle
            global_angle = robot_orientation + angle
            
            # Calculate beam direction vector
            beam_dir = np.array([
                np.cos(global_angle),
                np.sin(global_angle)
            ])
            
            # Check intersection with each obstacle
            for obstacle in obstacles:
                # Convert obstacle coordinates to meters
                obstacle_left = obstacle.left / 50.0
                obstacle_right = obstacle.right / 50.0
                obstacle_top = obstacle.top / 50.0
                obstacle_bottom = obstacle.bottom / 50.0
                
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
                        if distance < readings[i]:
                            readings[i] = distance

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