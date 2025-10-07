import heapq
import numpy as np
import shapely


class MidLevelPlanner:
    def __init__(self, feedback):
        self.feedback = feedback
        self.occupancy_grid = None
        self.map_resolution = None 
        # poses are 4x4 homogeneous transformation matrix
        self.map_origin = None  # <robot>/odom frame
        self.robot_pose = None  # <robot>/odom frame
        # high level plan is in <robot>/odom frame 
        self.feedback.print("INFO", "MidLevelPlanner initialized")

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
        
        ##### ----- debug code ----- #####
        # with open("/home/multyxu/adt4_output/debug_info.txt", "w") as debug_file:
        #     debug_file.write(f"map origin: {self.map_origin}\n")
        #     debug_file.write(f"pose in grid frame: {pose_in_grid_frame}\n")
        # np.save("/home/multyxu/adt4_output/grid_pose.npy", np.array([grid_i, grid_j]))
        ##### ----- debug code ----- #####
        
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
    
    def get_occupancy_range(self):
        # returns [xmin, xmax], [ymin, ymax] in grid cell coordinates
        minpt = self.grid_cell_to_global_pose((0, 0))
        maxpt = self.grid_cell_to_global_pose((-1, -1))
        return [minpt[0], maxpt[0]], [minpt[1], maxpt[1]]
    
    def plan_path(self, high_level_path, lookahead_distance = 1):
        '''
        Input: high level path in global frame, Nx2 numpy array
        Output: (bool, path) -> (success, path in odom frame)
        '''
        
        high_level_path_debug = high_level_path.copy()
        
        ## First get target point along the path
        # 1. project to current path distance
        # Assume self.robot_pose is in vision coordinate frame
        high_level_path = shapely.LineString(high_level_path[:, :2])
        current_point = shapely.Point(self.robot_pose[0, 3], self.robot_pose[1, 3])
        progress_distance = shapely.line_locate_point(high_level_path, current_point)
        progress_point = shapely.line_interpolate_point(high_level_path, progress_distance)
        # 2. get line point at lookahead
        target_distance = progress_distance + lookahead_distance
        target_point = shapely.line_interpolate_point(high_level_path, target_distance)
        target_4 = np.array([target_point.x, target_point.y, self.robot_pose[2, 3], 1]).reshape(4,1)
        target_cell = self.project_goal_to_grid(target_4)
        current_cell = self.global_pose_to_grid_cell(self.robot_pose[:, 3].reshape(4,1))
        
        ##### debug code #####
        # just publish the same path
        path = shapely.LineString(high_level_path_debug[:, :2])
        # project global point to local index
        local_path = [self.global_pose_to_grid_cell(np.array([pt[0], pt[1], 0, 1]).reshape(4,1)) for pt in high_level_path_debug[:, :2]]
        local_path = np.array(local_path)
        self.feedback.print("INFO", f"High level path: {high_level_path_debug.shape}, {high_level_path_debug}") 
        self.feedback.print("INFO", f"Local path cells: {local_path.shape}, {local_path}")
        np.save("/home/multyxu/adt4_output/high_level_path.npy", np.array(high_level_path_debug))
        np.save("/home/multyxu/adt4_output/local_path.npy", local_path)
        recovered_path = [self.grid_cell_to_global_pose((pt[0], pt[1])) for pt in local_path]
        recovered_path = np.array(recovered_path).reshape(-1,4)
        np.save("/home/multyxu/adt4_output/recovered_path.npy", recovered_path)
        
        # test astar path
        a_star_path = self.a_star(tuple(local_path[0]), (20,30))
        if a_star_path is not None:
            self.feedback.print("INFO", f"A* path cells: {a_star_path}")
            np.save("/home/multyxu/adt4_output/a_star_path.npy", np.array(a_star_path))
        
        # goal point
        np.save("/home/multyxu/adt4_output/target_cell.npy", np.array(target_cell))
        return True, path, high_level_path_debug[:, :2]
        ##### debug code #####
        
        local_path_cells = self.a_star(current_cell, target_cell)
        if local_path_cells is None:
            return False, None
        # self.feedback.print("INFO", f"Local path cells: {local_path_cells}")
        global_path = [tuple(self.grid_cell_to_global_pose(pt)[:2].flatten().tolist()) for pt in local_path_cells]
        local_path_wp = shapely.LineString(global_path)
        
        global_path_np = np.array([list(pt) for pt in global_path]).reshape(2,-1)
        self.feedback.print("INFO", f"Global path: {global_path_np}")
        return True, local_path_wp, global_path_np

    def project_goal_to_grid(self, goal):
        ## assumes that goal is in same coordinate frame as the occupancy grid
        goal_cell = self.global_pose_to_grid_cell(goal)
        # heurestic for right now will be clamping it to the occupancy grid.
        bx, by = self.get_occupancy_range()
        
        projected_cell = (
            max(bx[0], min(goal_cell[0], bx[1])).astype(int),
            max(by[0], min(goal_cell[1], by[1])).astype(int)
        )
        if self.occupancy_grid[projected_cell[0], projected_cell[1]] != 0:
            # if the goal is in an obstacle, find the nearest free cell
            free_cells = np.argwhere(self.occupancy_grid == 0)
            if free_cells.size == 0:
                return None
            dists = np.linalg.norm(free_cells - np.array(projected_cell).T, axis=1)
            projected_cell = tuple(free_cells[np.argmin(dists)])
        self.feedback.print("INFO", f"Projected cell: {projected_cell}")
        return projected_cell

    def a_star(self, start, goal):
        '''
        Generated A* path planning algorithm.
        Programed by Gemini 2.5 Pro
        '''
        if self.occupancy_grid is None:
            return None

        rows, cols = self.occupancy_grid.shape
        
        def is_valid(cell):
            return 0 <= cell[0] < rows and 0 <= cell[1] < cols

        def is_obstacle(cell):
            return self.occupancy_grid[cell[0], cell[1]] != 0

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
            self.feedback.print("INFO", f"Open set size: {len(open_set)}")
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-way connectivity
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
        
        ##### aryan #####
        # t, r = self.transform_lookup("<spot_vision_frame>", frame)
        # self.robot_grid_cell = self.global_pose_to_grid_cell([t.x, t.y, t.z, r])
