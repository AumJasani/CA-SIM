import pygame

class MapManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maps = {
            "default": {
                "start": [50, height // 2],
                "goal": [width - 50, height // 2],
                "obstacles": [
                    pygame.Rect(200, 100, 50, 400),
                    pygame.Rect(400, 0, 50, 300),
                    pygame.Rect(600, 300, 50, 300)
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
