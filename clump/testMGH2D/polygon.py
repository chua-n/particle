from skimage import draw, io
import numpy as np

imgSize = 64
img = np.zeros((imgSize, imgSize), dtype=np.uint8)
x = np.array([0, 0, 18, 18])
y = np.array([0, 32, 50, 0])
x += (imgSize - (x.max()-x.min()))//2
y += (imgSize - (y.max()-y.min()))//2
rr, cc = draw.polygon(x, y)
draw.set_color(img, [rr, cc], 255)
io.imsave("./testMGH2D/polygon.png", img)
io.imshow(img)
io.show()
