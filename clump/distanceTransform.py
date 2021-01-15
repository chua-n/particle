"""Generate centers and radii of the spheres that constitute a clump corresponding to a particle by distance transform method.

@author: chuan
"""
import numpy as np
from scipy import ndimage
from mayavi import mlab
from particle.pipeline import Sand


def run(cube: np.array):
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


if __name__ == "__main__":
    # data = np.load("./data/special/48.npy")
    # data = skimage.transform.rescale(data, 2)
    # data = data.reshape(1, 1, *data.shape)
    data = np.load("./data/test_set.npy")
    print(data.shape)
    for i in range(len(data)):
        cube = data[i, 0]
        c, r = run(cube)

        # 剔除最小的颗粒
        mask = r != r.min()
        c = c[mask]
        r = r[mask]

        mlab.points3d(*c.T, r*2, scale_factor=1,
                      resolution=30, mode="sphere")
        mlab.outline()
        mlab.axes()
        mlab.title(f"Total {len(r)} spheres.")

        sand = Sand(cube)
        sand.visualize(realistic=False)

        mlab.show()
