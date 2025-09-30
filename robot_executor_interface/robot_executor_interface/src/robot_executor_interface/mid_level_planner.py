import heapq
import numpy as np

class MidLevelPlanner:
    def __init__(self):
        self.occupancy_grid = None
        self.map_resolution = None 
        # poses are 4x4 homogeneous transformation matrix
        self.map_origin = None  # map frame
        self.robot_pose = None  # map frame
        # planed path should be in map frame
        # high level plan is in (probably) in map frame 

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
    
    def project_goal_to_grid(self, goal):
        pass
    
    def plan_path(self, high_level_path):
        '''
        Input: high level path in global frame, Nx2 numpy array
        Output: (bool, path) -> (success, path in odom frame)
        '''
        return False, None
    
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

    def set_grid(self, grid, resolution, origin):
        self.occupancy_grid = grid
        self.map_resolution = resolution
        self.map_origin = origin
    
    def set_robot_pose(self, pose):
        self.robot_pose = pose