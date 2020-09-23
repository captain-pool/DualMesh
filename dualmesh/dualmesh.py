#!/usr/bin/env python

import meshio
import numpy as np
import collections
cell_block = collections.namedtuple("CellBlock", ["type", "cells"])

# Source: https://codereview.stackexchange.com/questions/222623/pad-a-ragged-multidimensional-array-to-rectangular-shape
def get_dimensions(array, level=0):
    yield level, len(array)
    try:
        for row in array:
            yield from get_dimensions(row, level + 1)
    except TypeError: #not an iterable
        pass

def get_max_shape(array):
    dimensions = collections.defaultdict(int)
    for level, length in get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]

def iterate_nested_array(array, index=()):
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError: # final level
        yield (*index, slice(len(array))), array

def pad(array, fill_value):
    dimensions = get_max_shape(array)
    result = np.full(dimensions, fill_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result


def array_intersection(a, b):
    """Returns a boolean array of where b array's elements appear in the a array"""
    # Source: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)


def reorder_points(points):
    """Returns the order of given points, such that they are anticlockwise.

    Parameters:
        points:     numpy.ndarray
            Vertices of the polygon
    """
    assert isinstance(points, np.ndarray)
    assert points.shape[1] > 1
    # Calculate the center of the points
    c = points.mean(axis=0)
    # Calculate the angles between the horizontal and the line joining center to each point
    angles = np.arctan2(points[:,1] - c[1], points[:,0] - c[0])

    return np.argsort(angles).tolist()


def get_area(points):
    """Returns the area of a polygon given the vertices.

    Parameters:
        points:     numpy.ndarray
            Vertices of the polygon

    Warning: the points need to be ordered clockwise or anticlockwise."""

    # shift all the points by one
    shifted = np.roll(points, 1, axis=0)

    # Use the shoelace formula
    area = 0.5 * np.sum((shifted[:, 0] + points[:, 0])*(shifted[:, 1] - points[:, 1]))

    return np.abs(area)


def get_dual_points(mesh, index):
    """Returns the points of the dual mesh nearest to the point in the mesh given by the index.

    Parameters:
        mesh:       meshio.Mesh object
            Input mesh.

        index:      int
            Index of the point in the input mesh for which to calculate the nearest points of the dual mesh.

    """
    assert isinstance(mesh, meshio.Mesh)
    ## For each type of cell do the following
    # Find the cells where the given index appears
    cells = getattr(mesh, "rect_cells", mesh.cells)
    _idxs = [np.where(x[1] == index)[0] for x in cells]
    # Find the centers of all the cells
    _vs = []

    for i, x in enumerate(mesh.cells):
      points = []
      totp = 0
      totsum = []
      for idx in _idxs[i]:
        point = mesh.points[x[1][idx]]
        totsum.append(point[None, ...].mean(axis=1))
    #  if hasattr (mesh, "rect_cells"):
    #    breakpoint()
      _vs.append(np.vstack(totsum))
    #__vs = [mesh.points[x[1][_idxs[i]]].mean(axis=1) for i,x in enumerate(mesh.cells)]
    return np.concatenate(_vs, axis=0)


def get_dual(mesh, order=False):
    """Returns the dual mesh held in a dictionary with dual["points"] giving the coordinates and
    dual["cells"] giving the indicies of all the cells of the dual mesh.

    Parameters:
        mesh:       meshio.Mesh object
            Input mesh.

        order:      boolean
            Whether to reorder the indices of each cell, such that they are in anticlockwise order.
    """
    assert isinstance(mesh, meshio.Mesh)
    mesh_type = mesh.cells[0].type.lower()
    if mesh_type == "polygon":
      cells = [cell_block("polygon", pad(cell.data, -1)) for cell in mesh.cells]
      mesh.rect_cells = cells
    # Get the first set of points of the dual mesh
    new_points = get_dual_points(mesh, 0)
    vert_idxs = np.arange(len(new_points)).tolist()
    if order:
        new_order = reorder_points(new_points[vert_idxs])
        vert_idxs = np.array(vert_idxs)[new_order].tolist()

    # Create the containers for the points and the polygons of the dual mesh
    dual_points = new_points
    key = "polygon" if mesh_type != "polygon" else "triangle"
    dual_cells = {key: [vert_idxs]}

    for idx in range(1, len(mesh.points)):
        # Get the dual mesh points for a given mesh vertex
        new_points = get_dual_points(mesh, idx)
        # Find which of these new_points are already present in the dual_points
        inter = array_intersection(new_points, dual_points)

        # Add new_points to the dual mesh points that are not already there
        dual_points = np.concatenate((dual_points, new_points[~inter]))

        # Add the indices for the new cell to dual["cells"]
        inter = array_intersection(dual_points, new_points)
        vert_idxs = np.where(inter)[0].tolist()

        if order:
            # Reorder the indices, such that points are anticlockwise
            new_order = reorder_points(dual_points[vert_idxs])
            vert_idxs = np.array(vert_idxs)[new_order].tolist()

        dual_cells[key].append(vert_idxs)

    # Create the meshio mesh object
    dual = meshio.Mesh(dual_points, dual_cells)

    return dual
