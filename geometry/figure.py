"""Geomertical figure"""

__all__ = ['Circle', 'Parallelogram', 'Square', 'Triangle']

import base64
import io
import math
from collections.abc import Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

Point = NamedTuple('Point', [('x', int | float), ('y', int | float)])
Vector = Sequence[Point]
number = int | float


class BaseFigure(object):

    def __init__(self):
        self.x = 0
        self.y = 0
        self.angle = 45
        self.height = 0

    def get_area(self):
        pass

    @staticmethod
    def get_limits(coord: Vector) -> tuple[Point, Point]:
        x, y = zip(*coord)
        max_coord = max(max(x), max(y))
        gap = max_coord / 10
        lim = ((min(x) - gap, max(x) + gap),
               (min(y) - gap, max(y) + gap),
               )
        return lim

    @staticmethod
    def get_base64():
        pic_io_bytes = io.BytesIO()
        plt.savefig(pic_io_bytes, format='jpg')
        pic_io_bytes.seek(0)
        b64 = base64.b64encode(pic_io_bytes.read())
        pic_hash = b64.decode("utf-8").replace("\n", "")
        return pic_hash

    @property
    def radians(self):
        return math.radians(self.angle)


class Circle(BaseFigure):
    __slots__ = ['radius']

    def __init__(self, radius: number = 0):
        super().__init__()
        self.radius = radius

    def get_area(self):
        return math.pi * self.radius ** 2

    def plot(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        a = self.radius * np.cos(theta)
        b = self.radius * np.sin(theta)

        figure, axes = plt.subplots(1)
        axes.plot(a, b)
        axes.set_aspect('equal')
        plt.title(self.__class__.__name__)

        return self.get_base64()


class Cylinder(Circle):

    def __init__(self, radius: number = 0, height: number = 0):
        super().__init__(radius)
        self.height = height

    def get_volume(self) -> number:
        return self.get_area() * self.height


class Cone(Cylinder):

    def __init__(self, radius: number = 0, height: number = 0):
        super().__init__(radius, height)

    def get_volume(self) -> number:
        return super().get_volume() / 3


class Parallelogram(BaseFigure):
    __slots__ = ['a']

    def __init__(self, a: number = 0, b: number = 0, angle: number = 45):
        super().__init__()
        self.a = a
        self.b = b
        self.angle = angle

    def diagonal(self):
        cos_a_b = 2 * self.a * self.b * math.cos(self.radians)
        return math.sqrt(self.a ** 2 + self.b ** 2 - cos_a_b)

    def get_height(self):
        return self.b * math.sin(self.radians)

    def get_area(self):
        return self.a * self.b * math.sin(self.radians)

    def get_coord(self):
        h = self.get_height()
        leg = self.b * math.cos(self.radians)
        first = Point(0, 0)
        second = Point(self.a, 0)
        third = Point(self.a + leg, h)
        fourth = Point(leg, h)
        return first, second, third, fourth

    def plot(self):
        coord = self.get_coord()
        x_lim, y_lim = self.get_limits(coord)

        figure, axes = plt.subplots()
        axes.set(xlim=x_lim, ylim=y_lim)
        axes.add_patch(patches.Polygon(xy=coord, fill=False, color='green', linewidth=2))
        axes.grid()
        plt.title(self.__class__.__name__)

        return self.get_base64()


class Square(Parallelogram):
    def __init__(self, a: number = 0, **kwargs):
        super().__init__(a, a, angle=90)


class Triangle(Parallelogram):
    __slots__ = ['a', 'b', 'angle']

    def __init__(self, a: number = 0, b: number = 0, angle: number = 0):
        super().__init__(a, b, angle=angle)

    def perimeter(self):
        return self.a + self.b + self.diagonal()

    def get_area(self):
        return super().get_area() / 2

    def get_coord(self):
        coord = super(Triangle, self).get_coord()
        new_coord = tuple(xy for i, xy in enumerate(coord) if i != 2)
        return new_coord




class Trapeze(Triangle):

    def __init__(self, a: number = 0, b: number = 0, angle: number = 0):
        super().__init__(a, b, angle=angle)

    def c(self):
        h = self.get_height()
        triangle_short = math.sqrt(self.b ** 2 - h ** 2)
        return self.a - 2 * triangle_short

    def get_area(self):
        return ((self.a + self.c()) * self.get_height()) / 2


class Rhombus(BaseFigure):
    pass


if __name__ == '__main__':
    print(12)
