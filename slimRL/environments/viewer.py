import os

if "PYGAME_HIDE_SUPPORT_PROMPT" not in os.environ:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import time
import numpy as np


class Viewer:
    """
    Interface to pygame for visualizing mushroom native environments.

    """

    def __init__(
        self, env_width, env_height, width=500, height=500, background=(0, 0, 0)
    ):
        """
        Constructor.

        Args:
            env_width (float): The x dimension limit of the desired environment;
            env_height (float): The y dimension limit of the desired environment;
            width (int, 500): width of the environment window;
            height (int, 500): height of the environment window;
            background (tuple, (0, 0, 0)): background color of the screen.

        """
        self._size = (width, height)
        self._width = width
        self._height = height
        self._screen = None
        self._ratio = np.array([width / env_width, height / env_height])
        self._background = background

        self._initialized = False

    @property
    def screen(self):
        """
        Property.

        Returns:
            The screen created by this viewer.

        """
        if not self._initialized:
            pygame.init()
            self._initialized = True

        if self._screen is None:
            self._screen = pygame.display.set_mode(self._size)

        return self._screen

    @property
    def size(self):
        """
        Property.

        Returns:
            The size of the screen.

        """
        return self._size

    def line(self, start, end, color=(255, 255, 255), width=1):
        """
        Draw a line on the screen.

        Args:
            start (np.ndarray): starting point of the line;
            end (np.ndarray): end point of the line;
            color (tuple (255, 255, 255)): color of the line;
            width (int, 1): width of the line.

        """
        start = self._transform(start)
        end = self._transform(end)

        pygame.draw.line(self.screen, color, start, end, width)

    def polygon(self, center, angle, points, color=(255, 255, 255), width=0):
        """
        Draw a polygon on the screen and apply a roto-translation to it.

        Args:
            center (np.ndarray): the center of the polygon;
            angle (float): the rotation to apply to the polygon;
            points (list): the points of the polygon w.r.t. the center;
            color (tuple, (255, 255, 255)) : the color of the polygon;
            width (int, 0): the width of the polygon line, 0 to fill the
                polygon.

        """
        poly = list()

        for point in points:
            point = self._rotate(point, angle)
            point += center
            point = self._transform(point)
            poly.append(point)

        pygame.draw.polygon(self.screen, color, poly, width)

    def function(self, x_s, x_e, f, n_points=100, width=1, color=(255, 255, 255)):
        """
        Draw the graph of a function in the image.

        Args:
            x_s (float): starting x coordinate;
            x_e (float): final x coordinate;
            f (function): the function that maps x coorinates into y
                coordinates;
            n_points (int, 100): the number of segments used to approximate the
                function to draw;
            width (int, 1): thw width of the line drawn;
            color (tuple, (255,255,255)): the color of the line.

        """
        x = np.linspace(x_s, x_e, n_points)
        y = f(x)

        points = [self._transform([a, b]) for a, b in zip(x, y)]
        pygame.draw.lines(self.screen, color, False, points, width)

    @staticmethod
    def get_frame():
        """
        Getter.

        Returns:
            The current Pygame surface as an RGB array.

        """
        surf = pygame.display.get_surface()
        pygame_frame = pygame.surfarray.array3d(surf)
        frame = pygame_frame.swapaxes(0, 1)

        return frame

    def display(self, s):
        """
        Display current frame and initialize the next frame to the background
        color.

        Args:
            s: time to wait in visualization.

        """
        pygame.display.flip()
        time.sleep(s)

        self.screen.fill(self._background)

    def close(self):
        """
        Close the viewer, destroy the window.

        """
        self._screen = None
        pygame.display.quit()

    def _transform(self, p):
        return np.array(
            [p[0] * self._ratio[0], self._height - p[1] * self._ratio[1]]
        ).astype(int)

    @staticmethod
    def _rotate(p, theta):
        return np.array(
            [
                np.cos(theta) * p[0] - np.sin(theta) * p[1],
                np.sin(theta) * p[0] + np.cos(theta) * p[1],
            ]
        )
