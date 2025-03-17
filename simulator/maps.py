import pygame

class MapManager:
    def __init__(self):
        self.maps = {
            "default": {
                "conv": 50, # 50 pixels per meter
                "width": 800,
                "height": 600,
                "start": [50, 300],
                "goal": [750, 290],
                "obstacles": [
                    pygame.Rect(200, 100, 50, 400),
                    pygame.Rect(400, 0, 50, 300),
                    pygame.Rect(600, 300, 50, 300),
                    pygame.Rect(-1, 0, 1, 600), # left wall
                    pygame.Rect(0, -1, 800, 1), # top wall
                    pygame.Rect(0, 600, 800, 1), # bottom wall
                    pygame.Rect(800, 0, 1, 600) # right wall
                ]
            },
            "U": {
                "conv": 50, # 100 pixels per meter
                "width": 350,
                "height": 250,
                "start": [100, 50],
                "goal": [100, 200],
                "obstacles": [
                    pygame.Rect(0, 100, 200, 50)
                ]
            },
            "box": {
                "conv": 50, # 100 pixels per meter
                "width": 300,
                "height": 350,
                "start": [100, 50],
                "goal": [250, 300],
                "obstacles": [
                    pygame.Rect(30, 200, 140, 85)
                ]
            },
            "maze1": {
                "conv": 10, # 10 pixels per meter
                "width": 700,
                "height": 700,
                "start": [200, 200],
                "goal": [600, 600],
                "obstacles": [
                    pygame.Rect(290, 0, 20, 400)
                ]
            }
        }

    def load_map(self, map_name):
        return self.maps.get(map_name, self.maps["default"])

    def add_map(self, name, start, goal, obstacles):
        self.maps[name] = {
            "start": start,
            "goal": goal,
            "obstacles": obstacles
        }
