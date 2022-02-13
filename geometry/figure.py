__all__ = [
    'Circle', 'Parallelogram', 'Square', 'Triangle', 'Trapeze', 'Rhombus',
    'Sphere', 'Parallelepiped', 'Cube', 'Cylinder', 'Cone', 'Pyramid',
]

import base64
import io
import math
from collections.abc import Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from matplotlib import patches
from matplotlib import transforms
from matplotlib.axes import Axes

number = int | float
Point = NamedTuple('Point', [('x', number), ('y', number)])
Point3D = NamedTuple('Point', [('x', number), ('y', number), ('z', number)])
Vector = Sequence[Point]


class BaseFigure(object):

    def __init__(self):
        self.angle = 45

    def get_sub_coord(self):
        return Point(0, 0), 'None'

    def plot_annex(self, axes: Axes, deg: int = 0, delta_y: int = 0):
        coord, label, *zc = self.get_sub_coord()
        translate = self.translate(deg=deg, delta_y=delta_y)
        transform = translate + axes.transData
        axes.plot(*coord, color='red', label=label, transform=transform)
        plt.legend()

    def get_summary(self):
        pass

    @classmethod
    def get_figure(cls, projection=None, aspect='auto'):
        fig = plt.figure()
        axes = fig.add_subplot(111, projection=projection)
        plt.title(cls.__name__)
        axes.grid()
        axes.set_aspect(aspect)
        return fig, axes

    @staticmethod
    def annotate(text: str, axes: Axes, vector: Vector, va='top', ha='center'):
        xx, yy = zip(*vector)
        x = min(xx) + (max(xx) - min(xx)) / 2
        y = min(yy) + (max(yy) - min(yy)) / 2
        axes.annotate(text, xy=(x, y), fontsize=12, va=va, ha=ha)

    @staticmethod
    def get_limits(coord: Vector) -> tuple[Point, Point]:
        x, y = zip(*coord)
        max_coord = max(max(x), max(y))
        gap = max_coord / 9
        lim = ((min(x) - gap, max(x) + gap),
               (min(y) - gap, max(y) + gap),
               )
        return lim

    @staticmethod
    def translate(deg=0, delta_y=0):
        rotation = transforms.Affine2D().rotate_deg(deg)
        delta = transforms.Affine2D().translate(0, delta_y)
        return rotation + delta

    @staticmethod
    def get_base64():
        pic_io_bytes = io.BytesIO()
        plt.savefig(pic_io_bytes, dpi=130, format='jpg')
        pic_io_bytes.seek(0)
        b64 = base64.b64encode(pic_io_bytes.read())
        pic_hash = b64.decode('utf-8')
        return pic_hash

    @property
    def radians(self):
        return math.radians(self.angle)


class Circle(BaseFigure):
    __slots__ = ['radius']

    def __init__(self, radius: number = 5):
        super().__init__()
        self.radius = radius

    def get_area(self):
        return math.pi * self.radius ** 2

    def get_summary(self):
        return f'Площадь: {self.get_area():.2f}'

    @property
    def radius_coord(self):
        return tuple(zip(Point(0, 0), Point(self.radius, 0)))

    def plot_annex(self, axes: Axes, deg=0, delta_y=0):
        axes.plot(*self.radius_coord, color='red', label=f'radius {self.radius:.2f}')
        plt.legend()

    def get_coord(self):
        theta = np.linspace(0, 2 * np.pi, 50)
        a = self.radius * np.cos(theta)
        b = self.radius * np.sin(theta)
        return a, b

    def plot(self):
        figure, axes = self.get_figure(aspect='equal')
        axes.plot(*self.get_coord())
        self.plot_annex(axes)

        image = self.get_base64()
        plt.close(figure)

        return image


class Sphere(Circle):
    __slots__ = ['radius']

    def __init__(self, *args, **kwargs):
        super(Sphere, self).__init__(*args, **kwargs)

    def get_volume(self) -> number:
        return 4 * self.radius * self.get_area() / 3

    def get_summary(self):
        return f'Объём: {self.get_volume():.2f}'

    def get_coord(self):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = self.radius * np.cos(u) * np.sin(v)
        y = self.radius * np.sin(u) * np.sin(v)
        z = self.radius * np.cos(v)
        return x, y, z

    def plot(self, roof=True, wireframe=False):
        figure, axes = self.get_figure(projection='3d')
        xs, ys, zs = self.get_coord()

        axes.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        axes.plot_wireframe(xs, ys, zs)

        self.plot_annex(axes)
        image = self.get_base64()
        plt.close(figure)

        return image


class Cylinder(Circle):
    __slots__ = ['radius', 'height']

    def __init__(self, radius: number = 0, height: number = 0):
        super().__init__(radius)
        self.height = height

    def get_volume(self) -> number:
        return self.get_area() * self.height

    def get_summary(self):
        return f'Объём: {self.get_volume():.2f}'

    def get_coord(self):
        theta = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(0, self.height, 50)
        _, z_grid = np.meshgrid(theta, z)
        a, b = super().get_coord()
        return a, b, z_grid

    def plot(self, roof=True, wireframe=False):
        figure, axes = self.get_figure(projection='3d')

        xs, ys, zs = self.get_coord()

        axes.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        if roof:
            roof = patches.Circle((0, 0), radius=self.radius, alpha=0.6)
            axes.add_patch(roof)
            z = 0 if wireframe else np.ptp(zs)
            art3d.pathpatch_2d_to_3d(roof, z=z, zdir='z')
        if wireframe:
            axes.plot_wireframe(xs, ys, zs, alpha=0.5)
        else:
            axes.plot_surface(xs, ys, zs, alpha=0.5)

        self.plot_annex(axes)
        image = self.get_base64()
        plt.close(figure)

        return image


class Cone(Cylinder):

    def __init__(self, radius: number = 0, height: number = 0):
        super().__init__(radius, height)

    def get_volume(self) -> number:
        return super().get_volume() / 3

    def get_coord(self):
        theta = np.linspace(0, 2 * np.pi)
        r = np.linspace(0, self.radius)
        theta_grip, z_grip = np.meshgrid(theta, r)

        x = z_grip * np.cos(theta_grip)
        y = z_grip * np.sin(theta_grip)
        z = (np.sqrt(x ** 2 + y ** 2) / (self.radius / self.height))
        z = z[::-1]
        return x, y, z

    def plot(self, roof=False, wireframe=True):
        return super(Cone, self).plot(roof=True, wireframe=wireframe)


class Parallelogram(BaseFigure):
    __slots__ = ['a', 'b', 'angle']

    def __init__(self, a: number = 5, b: number = 4, angle: number = 45):
        super().__init__()
        self.a = a
        self.b = b
        self.angle = angle

    @property
    def diagonal(self):
        cos_a_b = 2 * self.a * self.b * math.cos(self.radians)
        return math.sqrt(self.a ** 2 + self.b ** 2 - cos_a_b)

    def get_height(self):
        return self.b * math.sin(self.radians)

    def get_area(self):
        return self.a * self.b * math.sin(self.radians)

    def get_summary(self):
        return f'Площадь: {self.get_area():.2f}'

    def get_leg(self):
        return self.b * math.cos(self.radians)

    def get_sub_coord(self):
        coord = self.get_coord()[3]
        return zip(coord, Point(coord.x, 0)), f'height {self.get_height():.2f}'

    def get_coord(self):
        h = self.get_height()
        leg = self.get_leg()
        first = Point(0, 0)
        second = Point(self.a, 0)
        third = Point(self.a + leg, h)
        fourth = Point(leg, h)
        return first, second, third, fourth

    def make_annotation(self, axes: Axes):
        coord = self.get_coord()
        self.annotate('b', axes, (coord[0], coord[-1]), va='bottom')
        self.annotate('a', axes, (coord[0], coord[1]), va='top')
        self.annotate('\u03B1', axes, (coord[0],), ha='left', va='bottom')

    def plot(self, deg=0, delta_y=0):
        coord = self.get_coord()
        x_lim, y_lim = self.get_limits(coord)

        figure, axes = self.get_figure(aspect='equal')
        axes.set(xlim=x_lim, ylim=y_lim)
        patch = patches.Polygon(xy=coord, fill=False, color='green', linewidth=2)

        self.plot_annex(axes, deg=deg, delta_y=delta_y)

        translate = self.translate(deg=deg, delta_y=delta_y)
        patch.set_transform(translate + axes.transData)
        axes.add_patch(patch)
        self.make_annotation(axes)

        image = self.get_base64()
        plt.close(figure)

        return image


class Parallelepiped(Parallelogram):
    __slots__ = ['a', 'b', 'angle', 'height']

    def __init__(self, a: number, b: number, angle: number, height: number):
        super().__init__(a, b, angle)
        self.height = height

    def get_volume(self):
        return self.get_area() * self.height

    def get_summary(self):
        return f'Объём: {self.get_volume():.2f}'

    def get_coord(self):
        coord = super(Parallelepiped, self).get_coord()
        coord = (*coord, coord[0])
        x = np.array([[crd.x] * 5 for crd in coord])
        y = np.array([[crd.y] * 5 for crd in coord])
        z = np.array([np.linspace(0, self.height, 5) for _ in range(5)])

        return x, y, z

    def plot(self, roof=True, wireframe=False):
        figure, axes = self.get_figure(projection='3d')

        coord = np.array(self.get_coord())

        roof_coord = super(Parallelepiped, self).get_coord()
        roof = patches.Polygon(xy=roof_coord, alpha=0.7)
        axes.add_patch(roof)
        art3d.pathpatch_2d_to_3d(roof, z=self.height, zdir='z')

        axes.set_box_aspect([np.ptp(axis) for axis in coord])
        axes.plot_surface(*coord, color='b', alpha=0.5)

        image = self.get_base64()
        plt.close(figure)

        return image


class Square(Parallelogram):
    __slots__ = ['a']

    def __init__(self, a: number = 5, **_):
        super().__init__(a, a, angle=90)

    def make_annotation(self, axes: Axes):
        coord = self.get_coord()
        self.annotate('a', axes, (coord[0], coord[1]), va='top')

    def get_sub_coord(self):
        coord = (self.get_coord()[3], self.get_coord()[1])
        return zip(*coord), f'diagonal {self.diagonal:.2f}'


class Cube(Parallelepiped):
    __slots__ = ['a']

    def __init__(self, a: number = 5, **_):
        super().__init__(a, a, 90, a)

    def get_volume(self) -> number:
        return self.get_area() * self.a

    def get_summary(self):
        return f'Объём: {self.get_volume():.2f}'


class Triangle(Parallelogram):

    def __init__(self, a: number = 5, b: number = 5, angle: number = 30):
        super().__init__(a, b, angle=angle)

    def perimeter(self):
        return self.a + self.b + self.diagonal

    def get_area(self):
        return super().get_area() / 2

    def get_sub_coord(self):
        bisect_angle = self.radians / 2
        bisect = (2 * self.a * self.b * math.cos(bisect_angle)) / (self.a + self.b)
        x = bisect * math.cos(bisect_angle)
        y = bisect * math.sin(bisect_angle)
        return zip(Point(0, 0), Point(x, y)), f'bisect {bisect:.2f}'

    def get_coord(self) -> Vector:
        coord = super(Triangle, self).get_coord()
        new_coord = tuple(xy for i, xy in enumerate(coord) if i != 2)
        return new_coord


class Trapeze(Parallelogram):
    __slots__ = ['a', 'c', 'height']

    def __init__(self, a: number = 5, c: number = 2, height: number = 5):
        self.c = c
        if a == c:
            super().__init__(a, c, angle=90)
        else:
            leg = (a - c) / 2
            angle = math.degrees(math.atan(height / leg))
            b = math.sqrt(leg ** 2 + height ** 2)
            super().__init__(a, b, angle=angle)

    def get_area(self):
        return ((self.a + self.c) * self.get_height()) / 2

    def make_annotation(self, axes: Axes):
        coord = self.get_coord()
        self.annotate('c', axes, (coord[2], coord[3]), va='bottom')
        self.annotate('a', axes, (coord[0], coord[1]), va='top')

    def get_sub_coord(self):
        middle_side = self.b / 2
        middle = (self.a + self.c) / 2
        mid_x = middle_side * math.cos(self.radians)
        mid_y = middle_side * math.sin(self.radians)
        middle_coord = (Point(mid_x, mid_y), Point(self.a - mid_x, mid_y))
        return zip(*middle_coord), f'middle line {middle:.2f}'

    def get_coord(self):
        coord = super(Trapeze, self).get_coord()
        third = Point(coord[2].x - 2 * self.get_leg(), coord[2].y)
        new_coord = (*coord[:2], third, coord[3])
        return new_coord


class Rhombus(Parallelogram):
    __slots__ = ['a', 'angle']

    def __init__(self, a: number = 5, angle: number = 15, **_):
        super().__init__(a, a, angle=angle)

    def get_diagonal(self):
        coord = self.get_coord()
        return np.linalg.norm(np.array(coord[2]) - np.array(coord[0]))

    def get_sub_coord(self):
        coord = self.get_coord()
        return zip(coord[0], coord[2]), f'diagonal {self.get_diagonal():.2f}'

    def get_area(self):
        return math.sin(self.radians) * self.a ** 2

    def make_annotation(self, axes: Axes):
        pass

    def plot(self, **_):
        rotation = -round(self.angle / 2)
        delta_y = round(self.diagonal / 2)
        return super(Rhombus, self).plot(deg=rotation, delta_y=delta_y)

    def get_limits(self, coord: Vector) -> tuple[Point, Point]:
        y = (0, self.diagonal)
        x = (0, math.sqrt(self.a ** 2 - (y[1] / 2) ** 2) * 2)
        return super().get_limits(tuple(zip(x, y)))


class Pyramid(Rhombus):
    __slots__ = ['a', 'angle', 'height']

    def __init__(self, a: number = 5, angle: number = 15, height: number = 10):
        self.height = height
        super().__init__(a, angle=angle)

    def get_coord(self):
        coord = super(Pyramid, self).get_coord()
        center_x = coord[2].x / 2
        center_y = center_x * math.tan(self.radians / 2)
        top = Point3D(center_x, center_y, self.height)
        base = [Point3D(p.x, p.y, 0) for p in coord]
        return *base, top

    def get_volume(self):
        return self.get_area() * self.height / 3

    def get_summary(self):
        return f'Объём: {self.get_volume():.2f}'

    def plot(self):
        figure, axes = self.get_figure(projection='3d')

        coord = np.array(self.get_coord())
        bases = (coord[:, 0], coord[:, 1], coord[:, 2])
        axes.scatter3D(*bases, marker='')
        axes.set_box_aspect([np.ptp(axis) for axis in bases])
        verts = [
            [coord[0], coord[1], coord[4]],
            [coord[0], coord[3], coord[4]],
            [coord[2], coord[1], coord[4]],
            [coord[2], coord[3], coord[4]]
        ]
        pyramid = art3d.Poly3DCollection(verts, linewidths=1, edgecolors='g', alpha=.5)
        axes.add_collection3d(pyramid)

        self.plot_annex(axes)
        image = self.get_base64()
        plt.close(figure)

        return image
