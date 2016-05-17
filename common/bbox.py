#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import warnings

import numpy


class BoundingBox(object):
    """Axis-aligned bounding box
    """

    @property
    def aspect(self):
        return self._dx / self._dy

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, val):
        self._dx = val

    @property
    def dy(self):
        return self._dy

    @dy.setter
    def dy(self, val):
        self._dy = val

    @property
    def size(self):
        return self._dx * self._dy

    def __init__(self, x, y, dx, dy, swap_xy=False, scale_factor=None):
        """A class for axis-aligned bounding boxes.

        Parameters
        ----------
        x: int or float
            x-coord of top-left box corner.
        y: int or float
            y-coord of top-left box corner.
        dx: int or float
            Size of box on x-axis.
        dy: int or float
            Size of box on y-axis.
        swap_xy: bool
            Whether to swap x and y axis (to convert from column-major axis order)
        scale_factor: None or float
            If not None used to scale the bounding box (=> bbox for subsampled
            image).
        """
        if swap_xy:
            self._x = y
            self._y = x
            self._dx = dy
            self._dy = dx
        else:
            self._x = x
            self._y = y
            self._dx = dx
            self._dy = dy
        if scale_factor:
            self.scale(scale_factor)

    @property
    def ndarray(self):
        """Get Numpy array containing x, y, delta x and delta y.

        Returns
        -------
        A Numpy ndarray containing x, y, delta x and delta y.
        """
        return numpy.array((self._x, self._y, self._dx, self._dy))

    def is_intersecting(self, other):
        """TODO: WRITEME
        """
        return (self._x < other.x + other.dx and other.x < self._x + self._dx and
                self._y < other.y + other.dy and other.y < self._y + self._dy)

    def construct_intersection(self, other):
        """Compute the axis-aligned bounding box from overlap of two AABBs.

        Parameters
        ----------
        bb1: iterable
            The first bounding box (specified by minx, miny, width, height)
        bb2: iterable
            The second bounding box, same format

        Returns
        -------
        A BoundingBox object, representing the intersection with the other box.
        """
        if not isinstance(other, BoundingBox):
            if hasattr(other, '__iter__'):
                try:
                    other = BoundingBox(*other)
                except:
                    print 'Could not convert other into BoundingBox'
            raise TypeError(
                ('other has to be of type BoundingBox or sequence containing 4'
                 'numbers'))

        if not self.is_intersecting(other):
            # no intersection
            raise ValueError('The bounding boxes do not intersect!')

        x = max(self._x, other.x)
        y = max(self._y, other.y)

        return BoundingBox(
            x, y,
            min(self._x + self._dx, other.x + other.dx) - x,
            min(self._y + self._dy, other.y + other.dy) - y)

    def __str__(self):
        return 'BoundingBox({0},{1},{2},{3})'.format(
            self._x, self._y, self._dx, self._dy
        )

    def __repr__(self):
        return 'BoundingBox({0},{1},{2},{3})'.format(
            self._x, self._y, self._dx, self._dy
        )

    def __eq__(self, other):
        arr1 = self.ndarray
        arr2 = other.ndarray
        if arr1.dtype == numpy.int and arr2.dtype == numpy.int:
            return all(arr1 == arr2)
        else:
            warnings.warn(('Performing check for almost-equality between '
                           'with numpy.allclose(), consider using int '
                           'bounding boxes.'))
            return numpy.allclose(arr1, arr2, rtol=1e-05, atol=1e-08)

    def __lt__(self, other):
        """Checks whether area is smaller than area of other bounding box
        """
        return self.size < other.size

    def draw(self, **kwargs):
        from matplotlib.patches import Rectangle
        # first index in numpy corresponds to rows, so swap x,y
        if 'fill' not in kwargs:
            kwargs['fill'] = False
        if 'color' not in kwargs:
            kwargs['color'] = 'r'
        return Rectangle(
            (self._y, self._x), self._dy, self._dx, **kwargs)

    def __add__(self, other):
        """Moves the box by a two-dimensional offset
        other: iterable
            offset on x and y axis
        """
        return BoundingBox(
            self._x + other[0], self._y + other[1],
            self._dx, self._dy)

    def __sub__(self, other):
        """Moves the box by a two-dimensional negative offset
        other: iterable
            offset on x and y axis
        """
        return BoundingBox(
            self._x - other[0], self._y - other[1],
            self._dx, self._dy)

    def scale(self, fac):
        """Scales the box (including x,y position!) by a factor in-place

        This can be used for adapting bounding boxes for downsampled
        images.

        Parameters
        ----------
        fac: int/float
        """
        assert fac != 0
        self._x *= fac
        self._y *= fac
        self._dx *= fac
        self._dy *= fac

    def scaled_copy(self, fac):
        """Returns a scaled copy of the bounding box

        Parameters
        ----------
        fac: int/float
            The scaling factor
        """
        rval = BoundingBox(self._x, self._y, self._dx, self._dy)
        rval.scale(fac)
        return rval

    def overlap(self, other):
        """Returns the intersection over union (IoU) measure for bbox pair
        """
        try:
            inter = self.construct_intersection(other)
            numerator = inter.size
        except ValueError:
            numerator = 0
        denominator = self.size + other.size - numerator
        return float(numerator) / denominator


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    bb1 = BoundingBox(0.5, 0.3, 0.6, 0.5)
    print 'bb1: {0}'.format(bb1, )
    bb2 = BoundingBox(0.4, 0.3, 0.2, 0.35)
    print 'bb2: {0}'.format(bb2, )
    bb3 = bb1.construct_intersection(bb2)
    print 'bb3: {0}'.format(bb3, )

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.add_patch(bb1.draw(facecolor='b'))
    ax.add_patch(bb2.draw(facecolor='r'))
    ax.add_patch(bb3.draw(facecolor='k'))
    plt.show()

    print 'IoU of {0} and {1}: {2}'.format(
        bb1, bb2, bb1.overlap(bb2))
    print 'IoU of {0} and {1}: {2}'.format(
        bb2, bb3, bb2.overlap(bb3))

# vim: set ts=4 sw=4 sts=4 expandtab:
