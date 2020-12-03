import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d
from mayavi import mlab

datafile = r"E:\Code\VAE\data\test_set.npy"
data = np.load(datafile)
sand = data[15, 0]
sand_skeleton = skeletonize(sand)
mlab.figure()
mlab.points3d(sand, color=(1, 1, 1))
mlab.figure()
mlab.points3d(sand_skeleton, color=(1, 1, 1))
mlab.show()
np.save('3.npy', sand)
