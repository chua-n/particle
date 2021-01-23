import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, draw

img = io.imread("./testMGH2D/polygon.png")
# io.imshow(img)
# io.show()

img_bf = ndimage.distance_transform_bf(img)
# io.imshow(img_bf, cmap="gray")
# io.show()

_img = img_bf.copy()
center, radii = [], []
while _img.max() > 0:
    row, col = np.unravel_index(np.argmax(_img, axis=None), img_bf.shape)
    center.append([row, col])
    radius = _img[row, col]
    radii.append(radius)
    rr, cc = draw.circle(row, col, round(radius))
    draw.set_color(_img, [rr, cc], 0)
    io.imshow(_img, cmap="gray")
    io.show()

np.random.seed(314)
center = np.array(center)
radii = np.array(radii)
fig, ax = plt.subplots()
ax.plot(*np.nonzero(img_bf.T), 'o')
tmp = center[:, 0].copy()
center[:, 0] = center[:, 1]
center[:, 1] = tmp
for c, r in zip(center, radii):
    ax.add_artist(plt.Circle(c, r, fill=False))
ax.set(xlim=((center[:, 0]-radii).min(), (center[:, 0]+radii).max()),
       ylim=((center[:, 1]-radii).min(), (center[:, 1]+radii).max()),
       title=f"Total {len(radii)} spheres.")
# plt.axis("equal")
ax.set_aspect(1)
plt.show()
