"""Generate centers and radii of the spheres that constitute a clump corresponding to a particle by distance transform method.

@author: chuan
"""
import numpy as np
from scipy import ndimage


def intersectionAngle(center1, r1, center2, r2):
    """计算两个圆/球的交角的余弦，根据余弦定理：$ cos \phi = (d^2 - r1^2 - r2^2) / 2*r1*r2 $。

    Return:
    -------
    cosPhi: 交角的余弦值。

    Note:
    -------
        1. 这里实际计算的“交角”是论文中定义的交角的补角，这样可以减少计算量；
        2. 由于给定的两个圆/球可能完全没有相交，此函数给出的返回值`cosPhi`可能大于1或小于-1，不过这无关紧要。
    """
    center1 = np.array(center1)
    center2 = np.array(center2)
    dSquared = np.sum((center1 - center2) ** 2)
    # 下面这一段代码无需使用
    # d = np.sqrt(dSquared)
    # if d < abs(r1 - r2):  # 内含关系
    #     return float('-inf')
    # elif d > r1 + r2:  # 外离关系
    #     return float('inf')
    cosPhi = (dSquared - r1**2 - r2**2) / (2*r1*r2)
    return cosPhi


def build(cube, lamda, phi):
    """Input a 3d image of a particle to build its clump.

    Parameters:
    -----------
    cube(np.array, 64*64*64): particle image to build clump
    lambda(float, [0, 1]): $\lambda$ in my dissertation, restrict the smallest balls
    phi(float, [0, 180]): $\phi$ in my dissertation, restrict the maximum overlap between balls
    """
    assert 0 <= lamda <= 1 and 0 <= phi <= 180
    cosPhi = np.cos(np.deg2rad(phi))
    distImg = ndimage.distance_transform_edt(cube)
    # 1. 距离图中全部候选球的坐标与半径
    allCoords = np.nonzero(distImg)
    allRadii = distImg[allCoords]
    allCoords = np.array(allCoords).T
    # 2. 按半径由大到小排序，并以最大球为筛选基准
    sortedIdx = np.argsort(allRadii)[::-1]
    rMax = allRadii[sortedIdx[0]]
    # 3. 筛选出待返回的clump球坐标及半径
    resCoords, resRadii = [], []
    for i in sortedIdx:
        if allRadii[i] / rMax >= lamda:
            flag = True
            if cosPhi != -1:
                for ctr, rds in zip(resCoords, resRadii):
                    cosAngle = intersectionAngle(allCoords[i], allRadii[i],
                                                 ctr, rds)
                    # 如果此候选球单元与某个现有球单元重叠量太大，舍弃此候选球
                    if cosAngle < cosPhi:
                        flag = False
                        break
            if flag:
                resCoords.append(allCoords[i])
                resRadii.append(allRadii[i])
    resCoords = np.array(resCoords)
    resRadii = np.array(resRadii)
    return resCoords, resRadii


def simpleBuild(cube: np.array):
    """The initial version of my clump-build algorithm, too simple to 
    get a smooth clump.
    """
    distImg = ndimage.distance_transform_edt(cube)
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
