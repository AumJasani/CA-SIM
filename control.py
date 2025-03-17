import numpy as np
from simulator import Simulator
from drr_controller import DRRController
from astar import AStar
import time
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Robot parameters
robot_radius = 0.3

# Create simulator instance
sim = Simulator(map_name="default", size=robot_radius, max_velocity=3, mass=8,
                 use_lidar=True, lidar_resolution=5.0, lidar_range_max=8.0,
                 plot_energies_graph=True, plot_velocities_graph=True)

# Define start and goal points (in meters)
start_point = sim.start_pos // sim.METER_TO_PIXEL
goal_point = sim.goal_pos // sim.METER_TO_PIXEL
print(f"Start point: {start_point}")
print(f"Goal point: {goal_point}")

# Create DRR controller
drr = DRRController(max_velocity=sim.max_velocity, robot_radius=robot_radius, 
                   time_step=1/60.0, recovery_time=1.0)  # 60 FPS simulation

# Initialize path planning variables
path_planner = AStar(grid_resolution=0.2, safety_margin=0.4, use_jps=True, debug=False)
waypoints = []
current_waypoint_idx = 0

# Map dimensions
map_width = sim.window_width // sim.METER_TO_PIXEL  # Width of the map in meters
map_height = sim.window_height // sim.METER_TO_PIXEL  # Height of the map in meters

# Variables to track robot trajectory
robot_trajectory = []
collision_points = []
point_cloud_history = []  # Add this line to store all point cloud data

# Flag for waypoint source (True for user-provided, False for A* generated)
use_user_waypoints = False  # Set this to True when using user-provided waypoints

# Variables for collision handling
collision_detected = False
collision_point = None
post_recovery_position = None
collision_time = 0
collision_to_world_rotation = np.eye(2)  # Identity rotation matrix

def plan_path_with_astar(start, goal, point_cloud):
    """Plan a path using A* from start to goal."""
    # Create grid representation of the environment
    grid = path_planner.create_grid(map_width, map_height, point_cloud, drr.robot_radius)
    
    # Find path using A* JPS
    path = path_planner.find_path(start, goal)
    
    if path and len(path) > 0:
        print(f"A* path found with {len(path)} waypoints")
        return [np.array(wp) for wp in path]
    else:
        print("A* failed to find a path")
        return None

def initialize_waypoints():
    """Initialize waypoints based on the selected source (user or A*)."""
    global waypoints, current_waypoint_idx
    
    if use_user_waypoints:
        # Use user-provided waypoints
        # This should be replaced with actual user waypoints
        waypoints = [
            start_point,
            # Add user waypoints here
            goal_point
        ]
    else:
        # Use A* to generate waypoints
        point_cloud = sim.get_latest_point_cloud()
        path = plan_path_with_astar(start_point, goal_point, point_cloud)
        if path:
            waypoints = path
        else:
            print("Failed to initialize waypoints")
            return False
    
    current_waypoint_idx = 0
    return True

def control_logic():
    """Main control logic for robot navigation."""
    global waypoints, current_waypoint_idx, collision_detected
    global collision_point, post_recovery_position, collision_time
    global robot_trajectory, collision_points, point_cloud_history
    
    # Get robot state
    robot_position = sim.circle_pos / sim.METER_TO_PIXEL
    robot_velocity = sim.velocity / sim.METER_TO_PIXEL
    robot_orientation = sim.orientation
    
    # Track robot trajectory and point cloud
    if len(robot_trajectory) == 0 or np.linalg.norm(robot_position - robot_trajectory[-1]) > 0.1:
        robot_trajectory.append(robot_position.copy())
        # Get and store point cloud data
        current_point_cloud = sim.get_latest_point_cloud()
        if current_point_cloud is not None and len(current_point_cloud) > 0:
            point_cloud_history.append(current_point_cloud)
    
    # Initialize waypoints if not already done
    if not waypoints:
        if not initialize_waypoints():
            return 0.0, 0.0, 0.0
    
    # Check for collision
    if sim.collision_occurred:
        if not collision_detected:  # Only process new collisions
            print("Collision detected! Applying recovery control...")
            collision_detected = True
            collision_time = time.time()
            
            # Get collision point
            collision_point = (sim.collision_point / sim.METER_TO_PIXEL 
                            if hasattr(sim, 'collision_point') else robot_position.copy())
            collision_points.append(collision_point.copy())
            
            # Apply DRR recovery controller
            next_waypoint = waypoints[current_waypoint_idx]
            time_interval = 1.0  # Time interval for recovery
            
            # First try-except for recovery control
            try:
                # Get recovery control
                u = drr.recovery_controller(
                    collision_point=collision_point,
                    next_waypoint=next_waypoint,
                    collision_time=collision_time,
                    time_interval=time_interval,
                    position_at_collision=robot_position,
                    collision_to_world_rotation=collision_to_world_rotation
                )
                
                # Set post-recovery position
                post_recovery_position = robot_position + np.array([u[0], u[1]]) * 0.5
                
            except Exception as e:
                print(f"Error in recovery control: {str(e)}")
                return 0.0, 0.0, 0.0
            
            # Apply waypoint adjustment
            point_cloud = sim.get_latest_point_cloud()
            adjusted_waypoints = drr.waypoint_adjustment(
                post_recovery_position=post_recovery_position,
                waypoint_list=waypoints,
                collision_to_world_rotation=collision_to_world_rotation,
                collision_segment_index=current_waypoint_idx,
                point_cloud=point_cloud,
                obstacle_radius=robot_radius/2
            )
            
            if adjusted_waypoints is not None and len(adjusted_waypoints) > 0:
                waypoints = adjusted_waypoints
                print(f"Waypoints adjusted: now have {len(waypoints)} waypoints")
                print(f"Waypoints: {waypoints}")
            
            return u[0], u[1], u[2]  # Return recovery control
        
    else:
        # Reset collision state if we've moved away from collision
        if collision_detected and collision_point is not None:
            if np.linalg.norm(robot_position - collision_point) > drr.robot_radius * 2:
                collision_detected = False
                collision_point = None
                post_recovery_position = None
    
    # Normal waypoint following
    if current_waypoint_idx < len(waypoints):
        current_waypoint = waypoints[current_waypoint_idx]
        
        # Check if waypoint reached
        distance = np.linalg.norm(robot_position - current_waypoint)
        if distance < 0.3:  # 30 cm threshold
            current_waypoint_idx += 1
            print(f"Reached waypoint {current_waypoint_idx-1}")
            
            # If we've reached the end of the path, stop
            if current_waypoint_idx >= len(waypoints):
                print("Reached final waypoint!")
                return 0.0, 0.0, 0.0
        
        # Calculate direction and velocity
        direction = current_waypoint - robot_position
        distance = np.linalg.norm(direction)
        
        # Normalize direction
        if distance > 0:
            direction = direction / distance
        
        # Use full velocity without scaling
        vx = direction[0] * sim.max_velocity
        vy = direction[1] * sim.max_velocity
        
        return vx, vy, 0.0
    
    return 0.0, 0.0, 0.0

def plot_waypoints_and_path():
    """
    Plot the waypoints, path, robot trajectory, and obstacles.
    This function is called when the simulation ends.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot map boundaries
    plt.plot([0, map_width, map_width, 0, 0], [0, 0, map_height, map_height, 0], 'k-', linewidth=2)
    
    # Plot waypoints and path
    if waypoints and len(waypoints) > 1:
        # Extract x and y coordinates
        x_coords = [wp[0] for wp in waypoints]
        y_coords = [wp[1] for wp in waypoints]
        
        # Plot waypoints as points
        plt.plot(x_coords, y_coords, 'bo', markersize=8, label='Waypoints')
        
        # Plot path as lines
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Planned Path')
        
        # Annotate waypoints with their indices
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            plt.annotate(f"{i}", (x, y), fontsize=10, ha='center', va='bottom')
    
    # Plot start and goal points
    plt.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    plt.plot(goal_point[0], goal_point[1], 'ro', markersize=10, label='Goal')
    
    # Plot robot trajectory if available
    if robot_trajectory and len(robot_trajectory) > 1:
        # Extract x and y coordinates
        x_coords = [pos[0] for pos in robot_trajectory]
        y_coords = [pos[1] for pos in robot_trajectory]
        
        # Plot trajectory as a line with gradient color to show direction
        plt.plot(x_coords, y_coords, 'g-', linewidth=1.5, alpha=0.6, label='Robot Trajectory')
        
        # Add arrow markers to show direction
        arrow_indices = np.linspace(0, len(x_coords)-1, min(20, len(x_coords))).astype(int)
        for i in arrow_indices[1:]:
            if i > 0 and i < len(x_coords):
                dx = x_coords[i] - x_coords[i-1]
                dy = y_coords[i] - y_coords[i-1]
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    dx, dy = dx/norm, dy/norm
                    plt.arrow(x_coords[i], y_coords[i], dx*0.2, dy*0.2, 
                             head_width=0.1, head_length=0.15, fc='g', ec='g', alpha=0.6)
    
    # Plot collision points if available
    if collision_points and len(collision_points) > 0:
        # Extract x and y coordinates
        x_coords = [point[0] for point in collision_points]
        y_coords = [point[1] for point in collision_points]
        
        # Plot collision points as red crosses
        plt.plot(x_coords, y_coords, 'rx', markersize=8, label='Collision Points')
    
    # Plot all obstacles from point cloud history
    try:
        if point_cloud_history and len(point_cloud_history) > 0:
            obstacles_x = []
            obstacles_y = []
            
            # Process all point clouds from history
            for point_cloud in point_cloud_history:
                # Check if point cloud is a list of dictionaries or coordinates
                if isinstance(point_cloud[0], dict):
                    for point_dict in point_cloud:
                        if 'position' in point_dict:
                            point_pos = point_dict['position']  # Keep in pixel coordinates
                            obstacles_x.append(point_pos[0])
                            obstacles_y.append(point_pos[1])
                        elif 'x' in point_dict and 'y' in point_dict:
                            obstacles_x.append(point_dict['x'])  # Keep in pixel coordinates
                            obstacles_y.append(point_dict['y'])
                else:
                    for point in point_cloud:
                        point_pos = np.array(point)  # Keep in pixel coordinates
                        obstacles_x.append(point_pos[0])
                        obstacles_y.append(point_pos[1])
            
            # Convert to meters only after collecting all points
            obstacles_x = np.array(obstacles_x)
            obstacles_y = np.array(obstacles_y)
            
            # Remove duplicate points to avoid overplotting
            unique_points = set(zip(obstacles_x, obstacles_y))
            obstacles_x, obstacles_y = zip(*unique_points)
            
            # Plot obstacles as small black dots with increased size and opacity
            plt.plot(obstacles_x, obstacles_y, "k.", markersize=1, alpha=0.7, label='Obstacles')
    except Exception as e:
        print(f"Error plotting obstacles: {e}")
    
    # Customize the plot
    plt.title('Robot Navigation Path and Trajectory')
    plt.xlabel('X coordinate (meters)')
    plt.ylabel('Y coordinate (meters)')
    plt.axis('equal')  # Equal aspect ratio
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Set axis limits with a bit of padding
    plt.xlim(-0.5, map_width + 0.5)
    plt.ylim(-0.5, map_height + 0.5)
    
    # Show the plot
    plt.tight_layout()
    
    # Save the plot with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"robot_path_trajectory_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as '{filename}'")
    
    plt.show()

# Run simulation
sim.run(control_logic)

# After simulation ends, plot the waypoints and path
if waypoints:
    plot_waypoints_and_path()
