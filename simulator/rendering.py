import pygame
import numpy as np

class Renderer:
    def __init__(self, width, height, colors):
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Navigate the Ball to the Goal")
        self.colors = colors

    def render_scene(self, circle_pos, circle_radius, goal_pos, goal_radius, 
                    obstacles, orientation, use_lidar=False, lidar_readings=None):
        """Render the complete scene including LiDAR if enabled"""
        # Clear screen
        self.screen.fill(self.colors["BLACK"])
        
        # Draw goal
        pygame.draw.circle(self.screen, self.colors["GREEN"], 
                         goal_pos.astype(int), goal_radius)
        
        # Draw obstacles
        for obstacle in obstacles:
            pygame.draw.rect(self.screen, self.colors["BLUE"], obstacle)
        
        # Draw LiDAR readings if enabled
        if use_lidar and lidar_readings is not None:
            angles = np.linspace(0, 2*np.pi, len(lidar_readings), endpoint=False)
            for angle, reading in zip(angles, lidar_readings):
                end_pos = circle_pos + np.array([
                    np.cos(orientation + angle) * reading,
                    np.sin(orientation + angle) * reading
                ])
                pygame.draw.line(
                    self.screen,
                    (100, 100, 100),  # Gray color for LiDAR beams
                    circle_pos.astype(int),
                    end_pos.astype(int),
                    1
                )
        
        # Draw robot and orientation line last (on top)
        pygame.draw.circle(self.screen, self.colors["RED"], 
                         circle_pos.astype(int), circle_radius)
        line_end = circle_pos + np.array([
            np.cos(orientation) * circle_radius,
            np.sin(orientation) * circle_radius
        ])
        pygame.draw.line(
            self.screen,
            self.colors["WHITE"],
            circle_pos.astype(int),
            line_end.astype(int),
            2
        )

        # Flip the screen vertically
        fliped = pygame.transform.flip(self.screen, False, True)
        self.screen.blit(fliped, (0, 0))
