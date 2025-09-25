class MidLevelPlanner:
    def __init__(self):
        self.occupancy_grid = None
        self.map_resolution = None 
        self.map_origin = None

    def plan_path(self, start, goal):
        # Implement path planning logic here
        pass

    def get_grid(self):
        # Implement logic to retrieve or generate a grid representation
        return self.occupancy_grid

    def set_grid(self, grid, resolution, origin):
        # Implement logic to set or update the grid representation
        self.occupancy_grid = grid
        self.map_resolution = resolution
        self.map_origin = origin