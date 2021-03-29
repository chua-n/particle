import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, draw

img = io.imread("/home/chuan/soil/particle/clump/testMGH2D/ellipse.png")
# io.imshow(img)
# io.show()

dist_img = ndimage.distance_transform_edt(img)
io.imshow(dist_img, cmap="gray")
io.show()
dist_img_ = dist_img.copy()

center, radii = [], []
while dist_img.max() > 0:
    row, col = np.unravel_index(np.argmax(dist_img, axis=None),
                                dist_img.shape)
    center.append([row, col])
    radius = dist_img[row, col]
    radii.append(radius)
    rr, cc = draw.circle(row, col, round(radius))
    draw.set_color(dist_img, [rr, cc], 0)
    io.imshow(dist_img, cmap="gray")
    io.show()

center = np.array(center)
radii = np.array(radii)
center = center[:, [1, 0]]
ax = plt.axes()
ax.plot(*np.nonzero(dist_img_.T), 'o', markersize=1)
for c, r in zip(center, radii):
    ax.add_artist(plt.Circle(c, r, fill=False))
ax.set(xlim=((center[:, 0]-radii).min(), (center[:, 0]+radii).max()),
       ylim=((center[:, 1]-radii).min(), (center[:, 1]+radii).max()),
       title=f"Total {len(radii)} spheres.")
# plt.axis("equal")
ax.set_aspect(1)
plt.show()
