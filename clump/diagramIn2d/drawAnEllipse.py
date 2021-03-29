from skimage import draw, io
import numpy as np

ratio = 2
imgSize = 64*ratio
img = np.zeros((imgSize, imgSize), dtype=np.uint8)
x = np.array([0, 0, 18, 18])*ratio
y = np.array([0, 32, 50, 0])*ratio
x += (imgSize - (x.max()-x.min()))//2
y += (imgSize - (y.max()-y.min()))//2
# rr, cc = draw.polygon(x, y)
rr, cc = draw.ellipse(imgSize/2, imgSize/2, imgSize/4, imgSize/3)
draw.set_color(img, [rr, cc], 255)
# io.imsave("./testMGH2D/polygon.png", img)
io.imsave("./ellipse.png", img)
io.imshow(img)
io.show()
