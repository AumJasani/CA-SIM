import numpy as np
import heapq

class AStar:
    def __init__(self, grid_resolution=0.2, safety_margin=0.3, use_jps=True, debug=False):
        """
        Initialize A* path planner
        
        Args:
            grid_resolution: Size of each grid cell in meters
            safety_margin: Additional clearance around obstacles in meters
            use_jps: Whether to use Jump Point Search optimization
            debug: Whether to print debugging information
        """
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        self.grid = None
        self.grid_origin = None
        self.grid_size = None
        self.use_jps = use_jps
        self.debug = debug
        self.jump_points_found = 0  # Counter for debug purposes
    
    def create_grid(self, width, height, point_cloud=None, robot_radius=0.2):
        """
        Create a grid representation of the environment.
        
        Args:
            width: Width of the environment in meters
            height: Height of the environment in meters
            point_cloud: LiDAR point cloud data representing obstacles
            robot_radius: Radius of the robot in meters
        
        Returns:
            A binary grid where 0 indicates free space and 1 indicates obstacles
        """
        # Calculate grid dimensions
        cols = int(np.ceil(width / self.grid_resolution))
        rows = int(np.ceil(height / self.grid_resolution))
        
        # Initialize grid with zeros (free space)
        self.grid = np.zeros((rows, cols), dtype=np.uint8)
        self.grid_size = (rows, cols)
        self.grid_origin = (0, 0)
        
        # If no point cloud is provided, return the empty grid
        if point_cloud is None or len(point_cloud) == 0:
            return self.grid
        
        # Mark cells as obstacles based on point cloud
        total_margin = self.safety_margin + robot_radius
        margin_cells = int(np.ceil(total_margin / self.grid_resolution))
        
        for point in point_cloud:
            # Convert point coordinates to grid indices
            x, y = point if isinstance(point, (list, tuple, np.ndarray)) else (point['x'], point['y'])
            grid_x = int(x / self.grid_resolution)
            grid_y = int(y / self.grid_resolution)
            
            # Skip if the point is outside the grid
            if not (0 <= grid_x < cols and 0 <= grid_y < rows):
                continue
            
            # Mark the cell and surrounding cells as obstacles
            for i in range(max(0, grid_y - margin_cells), min(rows, grid_y + margin_cells + 1)):
                for j in range(max(0, grid_x - margin_cells), min(cols, grid_x + margin_cells + 1)):
                    # Only mark if it's within the inflation radius
                    cell_center_x = (j + 0.5) * self.grid_resolution
                    cell_center_y = (i + 0.5) * self.grid_resolution
                    distance = np.sqrt((cell_center_x - x)**2 + (cell_center_y - y)**2)
                    if distance <= total_margin:
                        self.grid[i, j] = 1
        
        return self.grid
    
    def _get_neighbors(self, node):
        """
        Get the neighbors of a node in the grid.
        
        Args:
            node: Tuple of (row, col) indices
        
        Returns:
            List of neighboring nodes
        """
        row, col = node
        neighbors = []
        
        # If using JPS, we use a different neighbor pruning strategy
        if self.use_jps:
            # Get parent direction
            parent = getattr(self, 'parent', {}).get(node, None)
            if parent:
                # Calculate direction from parent to current
                dy = node[0] - parent[0]
                dx = node[1] - parent[1]
                
                # Normalize direction
                if dy != 0:
                    dy = dy // abs(dy)
                if dx != 0:
                    dx = dx // abs(dx)
                
                # Only consider "natural" neighbors in JPS
                # For diagonal movement, we have 3 neighbors
                if dx != 0 and dy != 0:
                    # Check diagonal neighbor
                    new_row, new_col = row + dy, col + dx
                    if self._is_valid(new_row, new_col):
                        neighbors.append(((new_row, new_col), np.sqrt(2)))
                    
                    # Check horizontal neighbor
                    new_row, new_col = row, col + dx
                    if self._is_valid(new_row, new_col):
                        neighbors.append(((new_row, new_col), 1))
                    
                    # Check vertical neighbor
                    new_row, new_col = row + dy, col
                    if self._is_valid(new_row, new_col):
                        neighbors.append(((new_row, new_col), 1))
                    
                    # Add forced neighbors
                    # Check if there's an obstacle next to us
                    if not self._is_valid(row - dy, col) and self._is_valid(row - dy, col + dx):
                        neighbors.append(((row - dy, col + dx), np.sqrt(2)))
                    if not self._is_valid(row, col - dx) and self._is_valid(row + dy, col - dx):
                        neighbors.append(((row + dy, col - dx), np.sqrt(2)))
                
                # For straight movement (horizontal), we have 1 neighbor
                elif dx != 0:
                    # Continue in same direction
                    new_row, new_col = row, col + dx
                    if self._is_valid(new_row, new_col):
                        neighbors.append(((new_row, new_col), 1))
                    
                    # Add forced neighbors
                    # Check if there's an obstacle above/below
                    if not self._is_valid(row - 1, col) and self._is_valid(row - 1, col + dx):
                        neighbors.append(((row - 1, col + dx), np.sqrt(2)))
                    if not self._is_valid(row + 1, col) and self._is_valid(row + 1, col + dx):
                        neighbors.append(((row + 1, col + dx), np.sqrt(2)))
                
                # For straight movement (vertical), we have 1 neighbor
                elif dy != 0:
                    # Continue in same direction
                    new_row, new_col = row + dy, col
                    if self._is_valid(new_row, new_col):
                        neighbors.append(((new_row, new_col), 1))
                    
                    # Add forced neighbors
                    # Check if there's an obstacle left/right
                    if not self._is_valid(row, col - 1) and self._is_valid(row + dy, col - 1):
                        neighbors.append(((row + dy, col - 1), np.sqrt(2)))
                    if not self._is_valid(row, col + 1) and self._is_valid(row + dy, col + 1):
                        neighbors.append(((row + dy, col + 1), np.sqrt(2)))
            else:
                # For the start node, consider all neighbors
                # Define the possible moves (8-connected grid)
                moves = [
                    (-1, 0),  # Up
                    (1, 0),   # Down
                    (0, -1),  # Left
                    (0, 1),   # Right
                    (-1, -1), # Up-Left
                    (-1, 1),  # Up-Right
                    (1, -1),  # Down-Left
                    (1, 1)    # Down-Right
                ]
                
                for dr, dc in moves:
                    new_row, new_col = row + dr, col + dc
                    if self._is_valid(new_row, new_col):
                        cost = np.sqrt(2) if dr != 0 and dc != 0 else 1
                        neighbors.append(((new_row, new_col), cost))
        else:
            # Standard A* neighbor finding (8-connected grid)
            moves = [
                (-1, 0),  # Up
                (1, 0),   # Down
                (0, -1),  # Left
                (0, 1),   # Right
                (-1, -1), # Up-Left
                (-1, 1),  # Up-Right
                (1, -1),  # Down-Left
                (1, 1)    # Down-Right
            ]
            
            for dr, dc in moves:
                new_row, new_col = row + dr, col + dc
                if self._is_valid(new_row, new_col):
                    cost = np.sqrt(2) if dr != 0 and dc != 0 else 1
                    neighbors.append(((new_row, new_col), cost))
        
        return neighbors
    
    def _is_valid(self, row, col):
        """Check if a cell is valid (within bounds and not an obstacle)."""
        return (0 <= row < self.grid_size[0] and 
                0 <= col < self.grid_size[1] and 
                self.grid[row, col] == 0)
    
    def _jump(self, node, direction):
        """
        Perform a jump in the given direction.
        
        Args:
            node: Current node (row, col)
            direction: Direction to jump (dy, dx)
        
        Returns:
            Jump point or None if not found
        """
        row, col = node
        dy, dx = direction
        
        # Check if current node is valid
        if not self._is_valid(row, col):
            return None
        
        # If we've reached the goal, return this node
        goal_node = getattr(self, 'goal_node', None)
        if goal_node and (row, col) == goal_node:
            if self.debug:
                print(f"JPS: Jump point at ({row}, {col}) - Goal reached")
            self.jump_points_found += 1
            return (row, col)
        
        # Check if this is a forced neighbor (a jump point)
        if dx != 0 and dy != 0:  # Diagonal movement
            # Check for horizontal/vertical obstacles and free diagonal cell
            if ((not self._is_valid(row - dy, col) and self._is_valid(row - dy, col + dx)) or
                (not self._is_valid(row, col - dx) and self._is_valid(row + dy, col - dx))):
                if self.debug:
                    print(f"JPS: Jump point at ({row}, {col}) - Forced neighbor (diagonal)")
                self.jump_points_found += 1
                return (row, col)
            
            # Try jumping horizontally and vertically
            if (self._jump((row + dy, col), (dy, 0)) is not None or
                self._jump((row, col + dx), (0, dx)) is not None):
                if self.debug:
                    print(f"JPS: Jump point at ({row}, {col}) - Expansion point (diagonal)")
                self.jump_points_found += 1
                return (row, col)
            
        else:  # Horizontal or vertical movement
            if dx != 0:  # Horizontal
                # Check for obstacles above/below
                if ((not self._is_valid(row - 1, col) and self._is_valid(row - 1, col + dx)) or
                    (not self._is_valid(row + 1, col) and self._is_valid(row + 1, col + dx))):
                    if self.debug:
                        print(f"JPS: Jump point at ({row}, {col}) - Forced neighbor (horizontal)")
                    self.jump_points_found += 1
                    return (row, col)
            else:  # Vertical
                # Check for obstacles left/right
                if ((not self._is_valid(row, col - 1) and self._is_valid(row + dy, col - 1)) or
                    (not self._is_valid(row, col + 1) and self._is_valid(row + dy, col + 1))):
                    if self.debug:
                        print(f"JPS: Jump point at ({row}, {col}) - Forced neighbor (vertical)")
                    self.jump_points_found += 1
                    return (row, col)
        
        # No jump point found, continue in same direction
        next_node = (row + dy, col + dx)
        if dx != 0 and dy != 0:  # Diagonal movement
            # For diagonal movement, we need to check both horizontal and vertical jumps
            diag_jump = self._jump(next_node, (dy, dx))
            if diag_jump:
                return diag_jump
        else:
            # For straight movement, just keep jumping
            return self._jump(next_node, (dy, dx))
        
        return None
    
    def _get_successors(self, node):
        """
        Get the successors of a node for JPS.
        
        Args:
            node: Current node (row, col)
        
        Returns:
            List of jump points with costs
        """
        successors = []
        neighbors = self._get_neighbors(node)
        
        if self.debug:
            print(f"JPS: Finding successors for node {node}")
        
        for neighbor, base_cost in neighbors:
            # Get direction
            dy = neighbor[0] - node[0]
            dx = neighbor[1] - node[1]
            
            # Perform jump in this direction
            jump_point = self._jump(neighbor, (dy, dx))
            if jump_point:
                # Calculate cost based on distance
                cost = np.sqrt((jump_point[0] - node[0])**2 + (jump_point[1] - node[1])**2)
                successors.append((jump_point, cost))
                if self.debug:
                    print(f"JPS: Added successor {jump_point} with cost {cost:.2f} from {node}")
        
        if self.debug and not successors:
            print(f"JPS: No successors found for {node}")
            
        return successors
    
    def _heuristic(self, node, goal):
        """
        Calculate the heuristic value (Euclidean distance) from node to goal.
        
        Args:
            node: Tuple of (row, col) indices
            goal: Tuple of (row, col) indices
        
        Returns:
            Heuristic value
        """
        return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
    
    def find_path(self, start_point, goal_point):
        """
        Find a path from start to goal using A* algorithm with Jump Point Search optimization.
        
        Args:
            start_point: Tuple of (x, y) coordinates in meters
            goal_point: Tuple of (x, y) coordinates in meters
        
        Returns:
            List of waypoints as (x, y) coordinates in meters
        """
        if self.grid is None:
            raise ValueError("Grid has not been created. Call create_grid() first.")
        
        # Reset debug counter
        self.jump_points_found = 0
        
        # Convert start and goal points to grid indices
        start_row = int(start_point[1] / self.grid_resolution)
        start_col = int(start_point[0] / self.grid_resolution)
        goal_row = int(goal_point[1] / self.grid_resolution)
        goal_col = int(goal_point[0] / self.grid_resolution)
        
        if self.debug:
            print(f"Planning path from ({start_row}, {start_col}) to ({goal_row}, {goal_col})")
            if self.use_jps:
                print("Using Jump Point Search optimization")
            else:
                print("Using standard A* algorithm")
        
        # Ensure start and goal are within grid boundaries
        # if not (0 <= start_row < self.grid_size[0] and 0 <= start_col < self.grid_size[1]):
        #     print(f"Start position {start_point} is outside the grid or in an obstacle.")
        #     return None
        
        # if not (0 <= goal_row < self.grid_size[0] and 0 <= goal_col < self.grid_size[1]):
        #     print(f"Goal position {goal_point} is outside the grid or in an obstacle.")
        #     return None
        
        # Check if start or goal are in obstacle cells
        # if self.grid[start_row, start_col] == 1:
        #     print("Start position is in an obstacle.")
        #     return None
        
        # if self.grid[goal_row, goal_col] == 1:
        #     print("Goal position is in an obstacle.")
        #     return None
        
        # Initialize the open and closed sets
        start_node = (start_row, start_col)
        goal_node = (goal_row, goal_col)
        self.goal_node = goal_node  # Store for _jump method
        
        open_set = []
        closed_set = set()
        
        # Store parent nodes for path reconstruction
        self.parent = {}
        
        # Use a dictionary to store the cost so far
        g_score = {start_node: 0}
        
        # f_score = g_score + heuristic
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        # Push the start node to the priority queue
        heapq.heappush(open_set, (f_score[start_node], id(start_node), start_node))
        
        while open_set:
            # Get the node with the lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # If we have reached the goal, reconstruct the path
            if current == goal_node:
                path = []
                while current in self.parent:
                    # Convert grid indices back to world coordinates
                    world_x = (current[1] + 0.5) * self.grid_resolution
                    world_y = (current[0] + 0.5) * self.grid_resolution
                    path.append((world_x, world_y))
                    current = self.parent[current]
                
                # Add the start position
                world_x = (start_node[1] + 0.5) * self.grid_resolution
                world_y = (start_node[0] + 0.5) * self.grid_resolution
                path.append((world_x, world_y))
                
                # Reverse the path to get start to goal
                path.reverse()
                
                # Smooth the path (optional)
                smoothed_path = self._smooth_path(path)
                
                # Clean up helper attributes
                delattr(self, 'goal_node')
                delattr(self, 'parent')
                
                # At the end of the method, if using JPS and in debug mode, print summary
                if self.use_jps and self.debug:
                    print(f"JPS: Found {self.jump_points_found} jump points during search")
                
                return smoothed_path
            
            # Add current to closed set
            closed_set.add(current)
            
            # Get successors based on whether we're using JPS or not
            if self.use_jps:
                successors = self._get_successors(current)
            else:
                successors = self._get_neighbors(current)
            
            # Explore neighbors or jump points
            for neighbor, move_cost in successors:
                # Skip if neighbor is in the closed set
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + move_cost
                
                # If neighbor is not in g_score or the new path is better
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update the path
                    self.parent[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal_node)
                    
                    # Add to open set if not already there
                    if neighbor not in [item[2] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))
        
        # If we get here, there is no path
        # Clean up helper attributes
        if hasattr(self, 'goal_node'):
            delattr(self, 'goal_node')
        if hasattr(self, 'parent'):
            delattr(self, 'parent')
            
        print("No path found from start to goal.")
        return None
    
    def _smooth_path(self, path, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001):
        """
        Smooth the path using gradient descent.
        
        Args:
            path: List of waypoints as (x, y) coordinates
            weight_data: Weight for the data term (higher keeps points closer to original)
            weight_smooth: Weight for the smoothing term (higher makes path smoother)
            tolerance: Convergence criterion
        
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        # Convert path to numpy array for easier manipulation
        path = np.array(path)
        
        # Create a copy of the path that we will smooth
        smoothed = path.copy()
        
        # Keep first and last points fixed
        change = tolerance + 1
        while change > tolerance:
            change = 0
            
            # Iterate through interior points
            for i in range(1, len(path) - 1):
                old_x, old_y = smoothed[i]
                
                # Update x coordinate
                smoothed[i][0] += weight_data * (path[i][0] - smoothed[i][0])
                smoothed[i][0] += weight_smooth * (smoothed[i+1][0] + smoothed[i-1][0] - 2 * smoothed[i][0])
                
                # Update y coordinate
                smoothed[i][1] += weight_data * (path[i][1] - smoothed[i][1])
                smoothed[i][1] += weight_smooth * (smoothed[i+1][1] + smoothed[i-1][1] - 2 * smoothed[i][1])
                
                # Calculate change
                change += abs(old_x - smoothed[i][0]) + abs(old_y - smoothed[i][1])
        
        return smoothed.tolist() 