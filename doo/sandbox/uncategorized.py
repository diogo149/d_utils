import scipy.spatial

from .. import utils


def convex_hull(points):
    # http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.spatial.ConvexHull.html
    if len(points) == 1:
        return points
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.points[hull.vertices]
    except scipy.spatial.qhull.QhullError:
        # example that errors: [[219, 389],[220, 388],[219, 389]]
        utils.info("Can't calculate convex hull on %s", points)
        return points


def polygon_area(corners):
    """
    calculated the area of a polygon given points on its surface

    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    https://en.wikipedia.org/wiki/Shoelace_formula
    """
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area
