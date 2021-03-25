"""Generate centers and radii of the spheres that constitute a clump corresponding to a particle by distance transform method.

@author: chuan
"""
import numpy as np
from scipy import ndimage


def oldRun(cube: np.array):
    """Input a 3d image of a particle to get its clump."""
    distImg = ndimage.distance_transform_bf(cube)
    pointCloud = np.array(np.nonzero(distImg)).T
    centers, radii = [], []
    while distImg.max() > 0:
        loc = np.unravel_index(np.argmax(distImg, axis=None), distImg.shape)
        centers.append(loc)
        radii.append(distImg[loc])
        mask = np.linalg.norm(pointCloud-loc, axis=1) <= radii[-1]
        locs = pointCloud[mask].T
        distImg[tuple(locs)] = 0
    centers = np.array(centers)
    radii = np.array(radii)
    return centers, radii


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


def run(cube, rho, phi):
    """Input a 3d image of a particle to get its clump."""
    distImg = ndimage.distance_transform_edt(cube)
    allCoords = np.nonzero(distImg)
    allRadii = distImg[allCoords]
    allCoords = np.array(allCoords).T
    # 按半径由大到小排序
    sortedIdx = np.argsort(allRadii)[::-1]
    # 待返回的clump球坐标及半径
    resCoords, resRadii = [], []
    rMax = allRadii[sortedIdx[0]]
    for i in sortedIdx:
        if allRadii[i] / rMax > rho:
            flag = True
            for ctr, rds in zip(resCoords, resRadii):
                angle = intersectionAngle(allCoords[i], allRadii[i], ctr, rds)
                if angle > phi:
                    flag = False
                    break
            if flag:
                resCoords.append(allCoords[i])
                resRadii.append(allRadii[i])
    resCoords = np.array(resCoords)
    resRadii = np.array(resRadii)
    return resCoords, resRadii


if __name__ == "__main__":
    from mayavi import mlab
    data = np.load("data/liutao/v1/particles.npz")["testSet"]
    print(data.shape)
    for i in range(10, len(data)):
        cube = data[i, 0]
        c, r = run(cube, 0.3, 150)

        print(f"Total {len(r)} spheres.")

        mlab.points3d(*c.T, r*2, scale_factor=1, opacity=1,
                      resolution=30, mode="sphere", color=(0.65,)*3)
        mlab.title(f"Total {len(r)} spheres.")
        mlab.outline()
        mlab.axes()
        mlab.show()
