import heapq
import numpy as np
import shapely


class MidLevelPlanner:
    def __init__(self, use_fake_path_planner, feedback):
        self.feedback = feedback
        self.occupancy_grid = None
        self.map_resolution = None 
        # poses are 4x4 homogeneous transformation matrix
        self.map_origin = None  # <robot>/odom frame
        self.robot_pose = None  # <robot>/odom frame
        # high level plan is in <robot>/odom frame 
        self.use_fake_path_planner = use_fake_path_planner
        self.feedback.print("INFO", f"MidLevelPlanner initialized, {self.use_fake_path_planner=}")

    # have a script to send the ActionSequenceMsg, check omniplanner
    def global_pose_to_grid_cell(self, pose):
        '''
        Input:
            - pose: (4,1) numpy array in map (global) frame
        Output:
            - (x, y): tuple in grid frame
        
        indexing: (i, j) = (row, col) = (y, x)
        '''
        pose_in_grid_frame = np.linalg.inv(self.map_origin) @ pose # (4,1)
        
        # Convert the pose to grid coordinates
        grid_j = int(pose_in_grid_frame[0, 0] / self.map_resolution)
        grid_i = int(pose_in_grid_frame[1, 0] / self.map_resolution)
                
        return (grid_i, grid_j)

    def grid_cell_to_global_pose(self, cell_indx):
        '''
        Input:
            - cell: (x, y) tuple in grid frame
        Output:
            - pose: (4,1) numpy array in map (global) frame
        indexing: (i, j) = (row, col) = (y, x)
        '''
        i, j = cell_indx
        pose_in_grid_frame = np.array([j * self.map_resolution, i * self.map_resolution, 0, 1]).reshape(4,1)
        pose_in_global_frame = self.map_origin @ pose_in_grid_frame
        return pose_in_global_frame
    
    
    def plan_path(self, high_level_path_metric, lookahead_distance_grid = 50):
        '''
        Input: high level path in global frame, Nx2 numpy array
        Output: (bool, path) -> (success, path in odom frame)
        '''
        
        high_level_path_debug = high_level_path_metric.copy()
        
        ## First get target point along the path
        # 1. project to current path distance
        # Assume self.robot_pose is in vision coordinate frame
        
        # convert poses to grid cells
        high_level_path_grid = [self.global_pose_to_grid_cell(np.array([pt[0], pt[1], 0, 1]).reshape(4,1)) for pt in high_level_path_metric[:, :2]]
        high_level_path_grid = np.array(high_level_path_grid).reshape(-1,2)
        current_point_grid = self.global_pose_to_grid_cell(self.robot_pose[:, 3].reshape(4,1))
        
        # convert poses to shapely points/lines
        high_level_path_shapely = shapely.LineString(high_level_path_grid)
        current_point_shapely = shapely.Point(current_point_grid[1], current_point_grid[0]) # (x,y) = (col, row)

        # where are we along the path?
        progress_distance_shapely = shapely.line_locate_point(high_level_path_shapely, current_point_shapely)
        progress_point_shapely = shapely.line_interpolate_point(high_level_path_shapely, progress_distance_shapely)
        
        # get line point at lookahead distance
        target_distance_shapely = progress_distance_shapely + lookahead_distance_grid
        target_point_shapely = shapely.line_interpolate_point(high_level_path_shapely, target_distance_shapely)
        
        # find the target in grid cell coordinates (make it free)
        target_point_grid = (int(target_point_shapely.x), int(target_point_shapely.y))
        target_point_grid_proj = self.project_goal_to_grid(target_point_grid)
        

        output = {
                    'target_point_metric': self.grid_cell_to_global_pose(target_point_grid_proj),
                    'path_shapely': shapely.LineString(high_level_path_metric[:, :2]),
                    'path_waypoints_metric': []
                }

        # plan using a_star
        a_star_path_grid = self.a_star(current_point_grid, target_point_grid_proj)
        


        if a_star_path_grid is None:
            # return False, None, None # or fallback
            # fall back using the high level path directly, and return empty visualization
            self.feedback.print("INFO", "A* failed, falling back to high level path")
            return False, output
        
        # convert a_star path to metric coordinates
        a_star_path_metric = [self.grid_cell_to_global_pose((pt[0], pt[1])) for pt in a_star_path_grid]
        a_star_path_metric = np.array(a_star_path_metric).reshape(-1,4)
        a_star_path_metric = a_star_path_metric[:, :2]
        a_star_path_execute = a_star_path_metric # only take first 30 points for now        

        # project global point to local index
        grid_path = [self.global_pose_to_grid_cell(np.array([pt[0], pt[1], 0, 1]).reshape(4,1)) for pt in high_level_path_debug[:, :2]]
        grid_path = np.array(grid_path)
        recovered_path = [self.grid_cell_to_global_pose((pt[0], pt[1])) for pt in grid_path]
        recovered_path = np.array(recovered_path).reshape(-1,4)    
        output['path_shapely'] = shapely.LineString(a_star_path_execute)
        output['path_waypoints_metric'] = a_star_path_metric
        return True, output

    def project_goal_to_grid(self, goal):
        ## assumes that goal is in same coordinate frame as the occupancy grid
        # goal_cell = self.global_pose_to_grid_cell(goal)
        # heurestic for right now will be clamping it to the occupancy grid.
        h, w = self.occupancy_grid.shape
        bound_i, bound_j = [0, h-1], [0, w-1]

        projected_cell = (
            max(bound_i[0], min(goal[0], bound_i[1])),
            max(bound_j[0], min(goal[1], bound_j[1]))
        )
        if self.occupancy_grid[projected_cell[0], projected_cell[1]] != 0:
            # if the goal is in an obstacle, find the nearest free cell
            free_cells = np.argwhere(self.occupancy_grid == 0)
            if free_cells.size == 0:
                return None
            dists = np.linalg.norm(free_cells - np.array(projected_cell).T, axis=1)
            projected_cell = tuple(free_cells[np.argmin(dists)])
        # self.feedback.print("INFO", f"Projected cell: {projected_cell}")
        return projected_cell

    def a_star(self, start, goal, diagonal_movement=True):
        '''
        Generated A* path planning algorithm.
        Programed by Gemini 2.5 Pro
        '''
        if diagonal_movement:
            connectivity = [(-1, 0), (1, 0), (0, -1), (0, 1),
                            (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-way connectivity
        else:
            connectivity = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-way connectivity
            
        if self.occupancy_grid is None:
            return None

        rows, cols = self.occupancy_grid.shape
        
        def is_valid(cell):
            return 0 <= cell[0] < rows and 0 <= cell[1] < cols

        def is_obstacle(cell):
            # return self.occupancy_grid[cell[0], cell[1]] != 0
            # -1 unknown, 100 occupied, 0 free
            return self.occupancy_grid[cell[0], cell[1]] > 0 # threshold

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        if not is_valid(start) or not is_valid(goal) or is_obstacle(start) or is_obstacle(goal):
            return None

        open_set = [(0, start)]  # (f_cost, position)
        came_from = {}
        g_cost = { (r, c): float('inf') for r in range(rows) for c in range(cols) }
        g_cost[start] = 0
        
        f_cost = { (r, c): float('inf') for r in range(rows) for c in range(cols) }
        f_cost[start] = heuristic(start, goal)

        open_set_hash = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in connectivity: # 4-way connectivity
                neighbor = (current[0] + dr, current[1] + dc)

                if not is_valid(neighbor) or is_obstacle(neighbor):
                    continue

                tentative_g_cost = g_cost[current] + 1
                if tentative_g_cost < g_cost[neighbor]:
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g_cost
                    f_cost[neighbor] = g_cost[neighbor] + heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_cost[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return None # No path found

    def get_grid(self):
        return self.occupancy_grid

    def set_grid(self, grid, resolution, origin, frame = None):
        self.occupancy_grid = grid
        self.map_resolution = resolution
        self.map_origin = origin
    
    def set_robot_pose(self, pose):
        self.robot_pose = pose
