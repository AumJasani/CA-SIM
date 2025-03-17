import numpy as np
import heapq
from astar import AStar

class DRRController:
    """
    Deformation Recovery and Replanning (DRR) Controller.
    Implements Algorithms 1, 2, 3, and 4 from the paper: Deformation Recovery Controller,
    Post-Impact Waypoint Adjustment, Collision-Inclusive Motion Primitive Generation,
    and Collision-inclusive Path Planning.
    """
    
    def __init__(self, max_velocity=5.0, robot_radius=0.2, time_step=0.1, recovery_time=1.0):
        """
        Initialize the DRR controller with basic parameters.
        
        Args:
            max_velocity: Maximum velocity of the robot
            robot_radius: Radius of the robot in meters
            time_step: Time step for simulation (delta t)
            recovery_time: Time for recovery in DRR (Tr)
        """
        # Controller parameters
        self.max_velocity = max_velocity
        self.robot_radius = robot_radius
        self.time_step = time_step
        self.recovery_time = recovery_time

    def recovery_controller(self, collision_point, next_waypoint, collision_time, 
                            time_interval, position_at_collision, collision_to_world_rotation):
        """
        Enhanced Algorithm 1: Deformation Recovery Controller
        
        This improved implementation provides better recovery behavior by:
        1. Adapting recovery strength based on collision severity
        2. Considering directional constraints in collision frame
        3. Adding small angular adjustments to help escape complex situations
        
        Args:
            collision_point: Position where collision occurred (world frame)
            next_waypoint: Next waypoint to reach (world frame)
            collision_time: Time when collision occurred
            time_interval: Time interval for the collision segment
            position_at_collision: Position at collision time (world frame)
            collision_to_world_rotation: Rotation matrix from collision to world frame
            
        Returns:
            numpy.ndarray: Control input u = [u_x, u_y, u_θ]
            
        Note:
            In a complete implementation following the paper:
            - velocity_at_collision would be used to determine impact behavior
            - body_to_world_rotation would be used to transform sensor readings from body frame
              to world frame (especially the Hall effect sensor readings mentioned in the paper)
        """
        # Step 2: Calculate target velocity in world frame
        w_v_T = (next_waypoint - position_at_collision) / max(time_interval - collision_time, 0.1)
        
        # Step 3: Transform to collision frame
        c_to_w_rotation = collision_to_world_rotation
        w_to_c_rotation = c_to_w_rotation.T  # Transpose for inverse rotation
        c_v_T = w_to_c_rotation @ w_v_T
        
        # Steps 4-6: Apply forward-only constraint with improvements
        # We use a softer constraint that allows slight backward motion if needed
        if c_v_T[0] < 0:
            # Instead of zeroing completely, allow small negative values for better maneuvering
            c_v_T[0] = max(c_v_T[0], -0.1 * self.max_velocity)
        
        # Steps 7-9: Apply velocity magnitude constraint
        v_norm = np.linalg.norm(c_v_T)
        if v_norm >= self.max_velocity:
            c_v_T = self.max_velocity * self._normalize(c_v_T)
        
        # Step 10: Calculate p0,x based on equation (5) with lb
        # In the paper, this calculation would use the displacement 
        # measured by Hall effect sensors (the body frame displacement l - l_s)
        
        # Calculate collision severity based on distance between current position and collision point
        current_to_collision_dist = np.linalg.norm(position_at_collision - collision_point)
        collision_severity = max(0.5, min(1.5, 1.0 / max(current_to_collision_dist, 0.1)))
        
        # Calculate deformation with adaptive scaling based on collision severity
        lb = self._calculate_deformation(collision_point) * collision_severity
        p0_x = lb
        
        # Step 11: Set p0,y to a small value to induce drift if needed
        # This helps break symmetry in recovery patterns
        collision_vector = collision_point - position_at_collision
        collision_vector_norm = np.linalg.norm(collision_vector)
        
        # Calculate a small lateral drift component based on collision geometry
        if collision_vector_norm > 0:
            # Use a small random value based on collision vector to break symmetry
            # The sign is determined by the y-component of the collision vector
            p0_y = 0.1 * self.robot_radius * np.sign(collision_vector[1]) 
        else:
            p0_y = 0.0
        
        # Steps 12-13: Calculate control inputs based on equations (4), (3), and (2)
        p0 = np.array([p0_x, p0_y])
        
        # Calculate control input in collision frame
        u_x, u_y = self._calculate_control_input(c_v_T, p0)
        
        # Calculate angular control input (u_θ)
        # Add small rotation to help escape tight situations
        # The sign is determined by the y-component of control to align with turning direction
        u_theta = self._calculate_angular_control(u_y)
        
        # Step 14: Combine control inputs
        u = np.array([u_x, u_y, u_theta])
        
        # Debug output
        print(f"Recovery controller analysis:")
        print(f"  Collision severity: {collision_severity:.2f}")
        print(f"  Deformation (p0): [{p0_x:.2f}, {p0_y:.2f}]")
        print(f"  Target velocity in collision frame: [{c_v_T[0]:.2f}, {c_v_T[1]:.2f}]")
        print(f"  Control output: [{u_x:.2f}, {u_y:.2f}, {u_theta:.2f}]")
        
        # Ensure the control is strong enough to overcome sticking
        min_recovery_magnitude = 0.1 * self.max_velocity
        control_magnitude = np.linalg.norm(u[:2])
        
        if control_magnitude < min_recovery_magnitude:
            # Scale up the control to ensure minimum recovery strength
            if control_magnitude > 0:
                scale_factor = min_recovery_magnitude / control_magnitude
                u[0] *= scale_factor
                u[1] *= scale_factor
                print(f"  Boosting recovery control by factor of {scale_factor:.2f}")
            else:
                # If control is zero, apply a default recovery in the x direction (away from obstacle)
                u[0] = min_recovery_magnitude
                print(f"  Applying default recovery control ({min_recovery_magnitude:.2f})")
        
        return u

    def waypoint_adjustment(self, post_recovery_position, waypoint_list, 
                            collision_to_world_rotation, collision_segment_index,
                            point_cloud=None, obstacle_radius=0.05):
        """
        Enhanced Algorithm 2: Post-Impact Waypoint Adjustment using A* JPS
        Finds nearest waypoint to current position and replans from there.
        """
        
        # Get the final goal waypoint
        goal_waypoint = waypoint_list[-1]
        
        # Find the nearest waypoint to post_recovery_position
        distances = [np.linalg.norm(np.array(post_recovery_position) - np.array(wp)) for wp in waypoint_list]
        nearest_waypoint_idx = np.argmin(distances)
        # If nearest waypoint is the last waypoint, use the previous one instead
        if nearest_waypoint_idx == len(waypoint_list) - 1 and len(waypoint_list) > 1:
            nearest_waypoint_idx = len(waypoint_list) - 2
            print(f"Nearest waypoint was final waypoint, using previous waypoint {nearest_waypoint_idx} instead")
        if collision_segment_index < nearest_waypoint_idx:
            nearest_waypoint_idx = collision_segment_index
        print(f"Found nearest waypoint {nearest_waypoint_idx} at distance {distances[nearest_waypoint_idx]:.2f}")
        
        # Initialize A* path planner with increased safety margins
        path_planner = AStar(
            grid_resolution=0.15,
            safety_margin=obstacle_radius,  # Increased safety margin
            use_jps=True,
            debug=False
        )
        
        # Calculate map dimensions with extra padding
        planning_radius = self.robot_radius * 20.0
        min_x = min(post_recovery_position[0], goal_waypoint[0]) - planning_radius
        max_x = max(post_recovery_position[0], goal_waypoint[0]) + planning_radius
        min_y = min(post_recovery_position[1], goal_waypoint[1]) - planning_radius
        max_y = max(post_recovery_position[1], goal_waypoint[1]) + planning_radius
        
        map_width = max_x - min_x
        map_height = max_y - min_y
        
        # Create grid representation for A* planning
        _ = path_planner.create_grid(map_width, map_height, point_cloud, self.robot_radius)
        
        # Plan new path from current position to goal
        print(f"Planning new path from {post_recovery_position} to goal {goal_waypoint}")
        new_path = path_planner.find_path(post_recovery_position, goal_waypoint)
        
        if new_path is None or len(new_path) == 0:
            print("Failed to find new path")
            return waypoint_list.copy()
        
        # Combine pre-collision waypoints with new path
        # Keep waypoints up to the nearest waypoint index
        combined_path = waypoint_list[:nearest_waypoint_idx]
        
        # Add the new path
        combined_path.extend([np.array(wp) for wp in new_path])
        
        print(f"Created combined path with {len(combined_path)} waypoints:")
        print(f"  - {nearest_waypoint_idx} original waypoints")
        print(f"  - {len(new_path)} new waypoints")
        
        return combined_path

    def get_motion_primitives(self, initial_state, motion_primitive_set, time_bound, obstacle_checker=None):
        """
        Algorithm 3: Collision-Inclusive Motion Primitive Generation
        
        Args:
            initial_state: Initial state s_d in free space
            motion_primitive_set: Set of motion primitives U_M
            time_bound: Upper bound of duration tau_f
            obstacle_checker: Function to check if a state is in collision (if None, all states are free)
            
        Returns:
            tuple: (reachable_set, costs_set, duration_set, collision_states_set)
                - reachable_set: Set of states reachable from initial_state in one step
                - costs_set: Set of costs for each primitive
                - duration_set: Set of durations for each primitive
                - collision_states_set: Set of collision states (1 if collision, 0 if not)
        """
        # Initialize sets as per algorithm (line 2)
        reachable_set = []
        costs_set = []
        duration_set = []
        collision_states_set = []
        
        # For each motion primitive (line 3)
        for primitive_idx, u_m in enumerate(motion_primitive_set):
            # Calculate edge e_m(t) according to equation (16) for t in [0, tau_f] (line 4)
            # This would involve integrating the dynamics with the given control input
            # For simplicity, we'll use a discrete-time approximation with Euler integration
            
            # Initialize variables for this primitive
            trajectory = [initial_state]
            collision_detected = False
            collision_time = None
            collision_state = None
            
            # Simulate the trajectory with time steps up to time_bound
            time_steps = int(time_bound / self.time_step)
            
            for t_idx in range(1, time_steps + 1):
                current_time = t_idx * self.time_step
                
                # Calculate next state by applying the motion primitive
                current_state = trajectory[-1]
                next_state = self._apply_motion_primitive(current_state, u_m, self.time_step)
                trajectory.append(next_state)
                
                # Check for collision (line 5)
                if obstacle_checker is not None and obstacle_checker(next_state):
                    collision_detected = True
                    collision_time = current_time
                    collision_state = next_state
                    break
            
            # Final state is either the last state in trajectory or collision state
            final_state = trajectory[-1]
            
            if not collision_detected:
                # Collision-free primitive (lines 6-13)
                zeta_m = 0  # No collision (line 6)
                tau_m = time_bound  # Full duration (line 7)
                s_d_m = final_state  # Final state (line 8)
                
                # Add to reachable set (line 9)
                reachable_set.append(s_d_m)
                
                # Calculate control effort cost J_D (line 10)
                # In a complete implementation, this would integrate ||u_m(t)||²
                # For simplicity, we'll use the norm of the primitive times duration
                J_D = np.linalg.norm(u_m) ** 2 * tau_m
                
                # Calculate total cost (line 11)
                # rho_t is a weight for time penalty
                rho_t = 1.0  # Example weight
                cost = J_D + rho_t * tau_m
                
                # Add to costs set (line 11)
                costs_set.append(cost)
                
                # Add to duration set (line 12)
                duration_set.append(tau_m)
                
                # Add to collision states set (line 13)
                collision_states_set.append(zeta_m)
                
            else:
                # Primitive with collision (lines 15-21)
                zeta_m = 1  # Collision detected (line 15)
                
                # In the paper, this would involve generating a post-collision state,
                # calculating collision costs, or pruning the primitive (line 16)
                # For simplicity, we'll use a basic model
                
                # Generate s_d_m, tau and calculate J_c (line 16)
                # s_d_m would be the post-collision state after recovery
                # tau would be time when collision occurred
                # J_c would be a collision cost
                
                tau = collision_time
                J_c = self._calculate_collision_cost(collision_state, u_m)
                
                # For now, use the same final state for simplicity
                # In a full implementation, this would use Algorithms 1, 2, and 4
                # to determine the post-collision state
                s_d_m = final_state
                
                # Add to reachable set (line 17)
                reachable_set.append(s_d_m)
                
                # Calculate control effort cost J_D (line 18)
                # For simplicity, we'll use the norm of the primitive times tau
                J_D = np.linalg.norm(u_m) ** 2 * tau
                
                # Calculate total cost (line 19)
                # Now including recovery time and collision cost
                rho_t = 1.0  # Example weight
                cost = J_D + rho_t * (tau + self.recovery_time) + J_c
                
                # Add to costs set (line 19)
                costs_set.append(cost)
                
                # Add to duration set (line 20)
                duration_set.append(tau_m)
                
                # Add to collision states set (line 21)
                collision_states_set.append(zeta_m)
        
        # Return all the sets (line 24)
        return reachable_set, costs_set, duration_set, collision_states_set
        
    def _apply_motion_primitive(self, state, motion_primitive, delta_t):
        """
        Apply a motion primitive to a state for one time step.
        
        Args:
            state: Current state (could be position/velocity/etc.)
            motion_primitive: Control input u_m
            delta_t: Time step
            
        Returns:
            new_state: State after applying the primitive
        """
        # In a complete implementation, this would apply the dynamics model
        # specific to the robot being used.
        # For a simple double integrator model:
        # position += velocity * delta_t
        # velocity += acceleration * delta_t
        
        # For simplicity, assume state is [x, y, theta, v_x, v_y, omega]
        # and motion_primitive is [a_x, a_y, alpha]
        
        # Create a copy of the state
        new_state = state.copy()
        
        # Update velocity components
        new_state[3] += motion_primitive[0] * delta_t  # v_x += a_x * dt
        new_state[4] += motion_primitive[1] * delta_t  # v_y += a_y * dt
        new_state[5] += motion_primitive[2] * delta_t  # omega += alpha * dt
        
        # Update position components
        new_state[0] += new_state[3] * delta_t  # x += v_x * dt
        new_state[1] += new_state[4] * delta_t  # y += v_y * dt
        new_state[2] += new_state[5] * delta_t  # theta += omega * dt
        
        return new_state
    
    def _calculate_collision_cost(self, collision_state, motion_primitive):
        """
        Calculate a cost for collision.
        
        Args:
            collision_state: State at collision
            motion_primitive: Control input that led to collision
            
        Returns:
            float: Collision cost
        """
        # In a complete implementation, this would calculate a cost based on:
        # - Severity of collision (e.g., velocity at impact)
        # - Direction of collision
        # - Type of obstacle
        
        # For simplicity, we'll use the norm of the velocity at collision
        # as a proxy for collision severity
        velocity_norm = np.linalg.norm(collision_state[3:5])  # Assuming state as [x, y, theta, v_x, v_y, omega]
        
        # Higher velocity = higher cost
        return 10.0 * velocity_norm  # Simple linear cost
        
    def _normalize(self, vector):
        """Normalize a vector."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _calculate_deformation(self, collision_point):
        """
        Calculate the deformation parameter based on the collision.
        This would implement equation (5) from the paper.
        
        In the full implementation, this would use the Hall effect sensor readings
        to measure the physical deformation of the robot body.
        
        For now, we use a simple approximation.
        """
        # Simplified model - in the real implementation this would use the Hall effect sensor data
        return self.robot_radius * 0.5
    
    def _calculate_control_input(self, target_velocity, deformation_point):
        """
        Calculate control inputs u_x and u_y based on equations (4) and (3).
        
        This would implement the actual control law from the paper.
        For simplicity, we'll use a basic proportional controller here.
        """
        # Gain parameters (would be tuned in the real implementation)
        kp_v = 1.0  # Gain for velocity
        kp_p = 0.5  # Gain for position
        
        # Calculate control inputs (simplified version)
        u_x = kp_v * target_velocity[0] - kp_p * deformation_point[0]
        u_y = kp_v * target_velocity[1] - kp_p * deformation_point[1]
        
        return u_x, u_y
    
    def _calculate_angular_control(self, lateral_control=0.0):
        """
        Calculate angular control input u_θ based on equation (2).
        
        Enhanced to add a small rotation based on lateral control direction
        to help the robot maneuver around obstacles.
        
        Args:
            lateral_control: The y-component of the control vector
            
        Returns:
            float: Angular control value
        """
        # Base rotation proportional to lateral control
        # This helps the robot turn in the direction it's trying to move laterally
        if abs(lateral_control) > 0.05:
            return 0.2 * np.sign(lateral_control)
        return 0.0

    def plan_path(self, start_state, goal_state, motion_primitive_set, obstacle_checker, 
                  max_iterations=1000, goal_threshold=0.5, rho_g=1.0, debug=False):
        """
        Algorithm 4: Collision-inclusive Path Planning
        
        Args:
            start_state: Initial state s_0
            goal_state: Goal state s_g
            motion_primitive_set: Set of motion primitives U_M
            obstacle_checker: Function to check if a state is in collision
            max_iterations: Maximum number of iterations for planning
            goal_threshold: Distance threshold to consider goal reached
            rho_g: Weight for goal heuristic in cost function
            debug: Whether to print debug information
            
        Returns:
            tuple: (path, controls, times, collision_flags, success)
                - path: List of states from start to goal
                - controls: List of control inputs applied at each step
                - times: List of durations for each step
                - collision_flags: List indicating if each step had a collision
                - success: Boolean indicating if planning succeeded
        """
        # Initialize open and closed sets (line 2)
        open_set = []  # Priority queue
        closed_set = set()  # Set of visited states (use tuples for hashable states)
        
        # Initialize dictionaries to track paths
        came_from = {}
        control_from = {}
        time_from = {}
        collision_from = {}
        
        # Initialize cost dictionary
        g_cost = {self._state_to_tuple(start_state): 0}
        
        # Calculate heuristic for start state
        start_cost = self._heuristic(start_state, goal_state) * rho_g
        
        # Add start state to open set
        heapq.heappush(open_set, (start_cost, self._state_to_tuple(start_state)))
        
        # Main planning loop (lines 3-22)
        iterations = 0
        success = False
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Pop state with lowest cost from open set (line 4)
            _, current_tuple = heapq.heappop(open_set)
            current_state = np.array(current_tuple)
            
            if debug and iterations % 100 == 0:
                print(f"Iteration {iterations}, Current state: {current_state}")
                print(f"Distance to goal: {np.linalg.norm(current_state[:2] - goal_state[:2])}")
            
            # Check if current state is close enough to goal (line 5)
            if self._is_goal_reached(current_state, goal_state, goal_threshold):
                success = True
                if debug:
                    print(f"Goal reached after {iterations} iterations!")
                break
            
            # Add current state to closed set (line 6)
            closed_set.add(current_tuple)
            
            # Get reachable states using motion primitives (line 8)
            reachable_states, costs, durations, collision_flags = self.get_motion_primitives(
                current_state, motion_primitive_set, self.time_step * 10, obstacle_checker
            )
            
            # For each successor state (lines 9-21)
            for i, next_state in enumerate(reachable_states):
                next_tuple = self._state_to_tuple(next_state)
                
                # Skip if already in closed set (line 10)
                if next_tuple in closed_set:
                    continue
                
                # Calculate tentative g_cost (line 11)
                tentative_g_cost = g_cost[current_tuple] + costs[i]
                
                # If this path is better or if state not in open set (lines 12-19)
                if next_tuple not in g_cost or tentative_g_cost < g_cost[next_tuple]:
                    # Update path information (lines 13-16)
                    came_from[next_tuple] = current_tuple
                    control_from[next_tuple] = motion_primitive_set[i]
                    time_from[next_tuple] = durations[i]
                    collision_from[next_tuple] = collision_flags[i]
                    
                    # Update cost (line 17)
                    g_cost[next_tuple] = tentative_g_cost
                    
                    # Calculate f_cost with heuristic (line 18)
                    f_cost = tentative_g_cost + self._heuristic(next_state, goal_state) * rho_g
                    
                    # Add to open set (line 19)
                    heapq.heappush(open_set, (f_cost, next_tuple))
        
        # Reconstruct path if goal was reached (line 23)
        if success:
            # Reconstruct the path
            path, controls, times, collision_flags = self._reconstruct_path(
                start_state, 
                goal_state if success else current_state,
                came_from, 
                control_from, 
                time_from,
                collision_from
            )
            
            return path, controls, times, collision_flags, True
        
        # Return empty result if planning failed
        if debug:
            print(f"Planning failed after {iterations} iterations")
        
        return [], [], [], [], False
    
    def _reconstruct_path(self, start_state, goal_state, came_from, control_from, time_from, collision_from):
        """
        Reconstruct the path from start to goal using the came_from dictionary.
        
        Args:
            start_state: Start state
            goal_state: Goal state
            came_from: Dictionary mapping each state to its predecessor
            control_from: Dictionary mapping each state to the control used to reach it
            time_from: Dictionary mapping each state to the time taken to reach it
            collision_from: Dictionary mapping each state to whether it involved a collision
            
        Returns:
            tuple: (path, controls, times, collision_flags)
        """
        path = [goal_state]
        controls = []
        times = []
        collision_flags = []
        
        # Get the state tuple for goal state
        current = self._state_to_tuple(goal_state)
        
        # Follow the path backwards from goal to start
        while current != self._state_to_tuple(start_state):
            prev = came_from.get(current)
            if prev is None:
                break
                
            # Add control, time, and collision flag
            controls.insert(0, control_from.get(current))
            times.insert(0, time_from.get(current))
            collision_flags.insert(0, collision_from.get(current))
            
            # Add state to path
            path.insert(0, np.array(prev))
            
            # Move to previous state
            current = prev
        
        # Add start state to beginning of path
        if path[0] is not start_state:
            path.insert(0, start_state)
        
        return path, controls, times, collision_flags
    
    def _state_to_tuple(self, state):
        """
        Convert state array to tuple for use as dictionary key.
        
        Args:
            state: State array
            
        Returns:
            tuple: State as tuple
        """
        return tuple(float(x) for x in state)
    
    def _is_goal_reached(self, current_state, goal_state, threshold):
        """
        Check if current state is close enough to goal state.
        
        Args:
            current_state: Current state
            goal_state: Goal state
            threshold: Distance threshold
            
        Returns:
            bool: True if goal is reached, False otherwise
        """
        # Check if position (x,y) is within threshold
        position_distance = np.linalg.norm(current_state[:2] - goal_state[:2])
        return position_distance < threshold
    
    def _heuristic(self, state, goal):
        """
        Calculate heuristic distance from state to goal.
        
        Args:
            state: Current state
            goal: Goal state
            
        Returns:
            float: Heuristic distance
        """
        # Euclidean distance between positions (x,y)
        return np.linalg.norm(state[:2] - goal[:2])
