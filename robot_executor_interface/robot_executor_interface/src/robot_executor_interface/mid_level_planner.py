import heapq
import numpy as np


class MidLevelPlanner:
    def __init__(self, transform_lookup):
        self.occupancy_grid = None
        self.map_resolution = None
        self.map_origin = None  # odom frame?
        self.robot_pose = None  # odom
        self.transform_lookup = transform_lookup
        # planed path should be in odom frame
        # high level plan is in (probably) in map frame, but trasnformed to odom

    # have a script to send the ActionSequenceMsg, check omniplanner
    def global_pose_to_grid_cell(self, pose):
        pass

    def grid_cell_to_global_pose(self, cell):
        pass
    
    def get_occupancy_range(self):
        # returns [xmin, xmax], [ymin, ymax] in grid cell coordinates
        pass

    def project_goal_to_grid(self, goal):
        ## assumes that goal is in same coordinate frame as the occupancy grid
        goal_cell = self.global_pose_to_grid_cell(goal)
        # heurestic for right now will be clamping it to the occupancy grid.
        bx, by = self.get_occupancy_range()
        goal_cell[0] = max(bx[0], min(goal_cell[0], bx[1]))
        goal_cell[1] = max(by[0], min(goal_cell[1], by[1]))
        if self.occupancy_grid[goal_cell[0], goal_cell[1]] != 0:
            # if the goal is in an obstacle, find the nearest free cell
            free_cells = np.argwhere(self.occupancy_grid == 0)
            if free_cells.size == 0:
                return None
            dists = np.linalg.norm(free_cells - np.array(goal_cell), axis=1)
            goal_cell = tuple(free_cells[np.argmin(dists)])
        return goal_cell

    def plan_path(self, start, goal):
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

    def set_grid(self, grid, resolution, origin, frame = None):
        self.occupancy_grid = grid
        self.map_resolution = resolution
        self.map_origin = origin
        t, r = self.transform_lookup("<spot_vision_frame>", frame)
        self.robot_grid_cell = self.global_pose_to_grid_cell([t.x, t.y, t.z, r])