import random
import numpy as np
from mayavi import mlab
import scipy
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

# from min_bound_sphere.exact_min_bound_sphere_3D import exact_min_bound_sphere_3D
from particle.utils import Circumsphere
from particle.pipeline import Sand

mlab.options.offsceen = True


def getDelaunayTetrahedralMesh(sandCube):
    sand = Sand(sandCube)
    convexHull = sand.sand_convex_hull()
    # pointCloud = sand.point_cloud().T
    # pointCloud, _ = sand.surface()
    pointCloud = convexHull.points
    mesh = Delaunay(pointCloud)
    # mesh = Delaunay(pointCloud, furthest_site=True)  # 不太行
    # mesh = Delaunay(pointCloud, incremental=True)
    # mesh = Delaunay(pointCloud, furthest_site=True, incremental=True)  # 不太行
    return mesh


def circumscribedSphere(tetrahedron):
    """问题：
    1、可能出现QhullError
    2、计算出来的半径可能会无穷大；"""
    radius, center, *_ = exact_min_bound_sphere_3D(tetrahedron)
    # 暂时这样解决一下半径无穷大问题
    # 这样处理不好，无穷大也是一种信息，最好在后期处理的时候使用异常捕捉
    if radius == np.inf:
        radius = 0
        center = (0, 0, 0)
    return center, radius


def intersectionAngle(center1, r1, center2, r2):
    """cos \phi = (d^2 - r1^2 - r2^2) / 2*r1*r2

    Return:
    -------
    phi: 角度，角度制
    """
    center1 = np.array(center1)
    center2 = np.array(center2)
    dSquare = np.sum((center1 - center2) ** 2)
    # RuntimeWarning: invalid value encountered in arccos
    cosPhi = (dSquare - r1**2 - r2**2) / (2*r1*r2)
    phi = np.arccos(cosPhi)
    phi *= (180 / np.pi)
    return phi


class Plotter:
    random.seed(3.14)

    @classmethod
    def randomColor(cls):
        color = (random.random(), random.random(), random.random())
        return color

    @classmethod
    def plotSphere(cls, center, radius, nPoints=100, opacity=1.0):
        """Draw a sphere according to given center and radius.

        Parameters:
        -----------
        center(tuple): (x, y, z) coordinate
        radius(float): radius of the sphere
        """
        u = np.linspace(0, 2 * np.pi, nPoints)
        v = np.linspace(0, np.pi, nPoints)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        # scene = mlab.points3d(x, y, z, mode="point")
        scene = mlab.mesh(x, y, z, color=cls.randomColor(), opacity=opacity)
        return scene

    @classmethod
    def plotTetrahedron(cls, tetrahedron, opacity=1.0):
        """Tetrahedron: tri.points[tri.simplices[i]]

        Delaunay tetrahedral似乎不是根据三维体画出来的，而是三维表面画出来的。"""
        scene = mlab.triangular_mesh(tetrahedron[:, 0], tetrahedron[:, 1], tetrahedron[:, 2],
                                     [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
                                     color=cls.randomColor(), opacity=opacity)
        return scene

    @classmethod
    def plotIntegration(cls, sandCube, tetrahedron):
        sand = Sand(sandCube)
        scene = sand.visualize(realistic=False, opacity=0.6)
        Plotter.plotTetrahedron(tetrahedron)
        center, radius = circumscribedSphere(tetrahedron)
        Plotter.plotSphere(center, radius, opacity=0.6)
        return scene


def filtSphere(tetrahedralMesh, rho, phi):
    nTetra = tetrahedralMesh.simplices.shape[0]
    centers = np.full((nTetra, 3), -1, dtype=float)
    radii = np.full(nTetra, -1, dtype=float)
    for i in range(nTetra):
        try:
            tetra = tetrahedralMesh.points[tetrahedralMesh.simplices[i]]
            centers[i], radii[i] = circumscribedSphere(tetra)
        except QhullError:
            print(f"Encountering QhullError in index-{i}")
            continue
    sortedIndex = np.argsort(radii)[::-1]
    remainedSphereIndex = []
    rMax = radii[sortedIndex[6000]]
    for i in range(6000, nTetra):
        r = radii[sortedIndex[i]]
        if r / rMax > rho:
            flag = True
            for ind in remainedSphereIndex:
                angle = intersectionAngle(centers[ind], radii[ind],
                                          centers[sortedIndex[i]], r)
                if angle > phi:
                    flag = False
                    break
            if flag:
                remainedSphereIndex.append(sortedIndex[i])
    remainedCenters = centers[remainedSphereIndex]
    remainedRadii = radii[remainedSphereIndex]
    return remainedCenters, remainedRadii


if __name__ == "__main__":
    datafile = r"E:\Code\VAE\data\test_set.npy"
    data = np.load(datafile)
    sandCube = data[20, 0]
    # sandCube = np.load('132.npy')
    mesh = getDelaunayTetrahedralMesh(sandCube)

    def plotResult():
        # mlab.figure(bgcolor=(1, 1, 1))
        Sand(sandCube).visualize(realistic=False)
        remainedCenters, remainedRadii = filtSphere(mesh, 0.4, 120)
        for center, r in zip(remainedCenters, remainedRadii):
            Plotter.plotSphere(center, r)
        mlab.show()

    def testDelaunay(tetrahedralMesh, step=100):
        mlab.figure(bgcolor=(1, 1, 1))
        for i in range(0, tetrahedralMesh.simplices.shape[0], step):
            Plotter.plotTetrahedron(
                tetrahedralMesh.points[tetrahedralMesh.simplices[i]])
        mlab.show()
    # plotResult()
    # testDelaunay(mesh, 10)
    tetrahedralMesh = getDelaunayTetrahedralMesh(sandCube)
    nTetra = tetrahedralMesh.simplices.shape[0]
    for i in range(nTetra):
        tetra = tetrahedralMesh.points[tetrahedralMesh.simplices[i]]
        Plotter.plotIntegration(sandCube, tetra)
        mlab.show()
