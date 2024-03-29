import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from skimage.measure import marching_cubes
import torch


def fig2array(fig):
    """Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it

    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values

    Note: Use fig.canvastostring_argb() to get the alpha channel of an image if you want.
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    # buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(h, w, 3)
    return buf


def makeGrid(images: Union[np.ndarray, List[np.ndarray]],
             filename: str,
             nrow: int = 8,
             normalize: bool = True):
    """Make a grid of images from input `images`.

    Parameters:
    -----------
    images: a batch of images whose shape is (H, W, C)
    filename: the name of the image-grid file to be saved
    nrow (int, optional): Number of images displayed in each row of the grid
    normalize (bool, optional): If True, shift the image to the range (0, 1),
        by the min and max values specified by :attr:`range`. Default: ``True``.
    """
    from torchvision.utils import save_image

    try:
        # 对于numpy的一些图片数组，torch.from_numpy(image) 会发生异常————PyTorch的奇哉怪也
        torch.from_numpy(images[0])
    except ValueError:
        images = images.copy() if isinstance(images, np.ndarray) else [
            img.copy() for img in images]

    # get the batch, height, width, channel
    b = len(images)
    h, w, c = images[0].shape
    tensors = torch.empty((b, c, h, w), dtype=torch.float32)
    for i, image in enumerate(images):
        for j in range(c):
            tensors[i, j] = torch.from_numpy(image[:, :, j])

    save_image(tensors, filename, nrow=nrow, normalize=normalize)
    return


def singleSphere(center, radius, nPoints=100, opacity=1.0, color=None):
    """Draw a sphere according to given center and radius.

    Parameters:
    -----------
    center(tuple): (x, y, z) coordinate
    radius(float): radius of the sphere
    """
    if color is None:
        random.seed(3.14)
        color = (random.random(), random.random(), random.random())
    u = np.linspace(0, 2 * np.pi, nPoints)
    v = np.linspace(0, np.pi, nPoints)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    from mayavi import mlab
    # scene = mlab.points3d(x, y, z, mode="point")
    scene = mlab.mesh(x, y, z, color=color, opacity=opacity)
    return scene


def sphere(center, radius, resolution=30, **kwargs):
    """Draw some spheres according to given center and radius.

    Parameters:
    -----------
    center(np.array, n*3): x, y, z coordinates of n spheres
    radius(np.array, n): radii of the n spheres
    resolution(int): resolution of each sphere in returned scene
    """
    x, y, z = center[:, 0], center[:, 1], center[:, 2]
    from mayavi import mlab
    scene = mlab.points3d(x, y, z, radius*2, scale_factor=1,
                          resolution=resolution, **kwargs)
    return scene


def tetrahedron(tetrahedron, opacity=1.0, color=None):
    """Tetrahedron: tri.points[tri.simplices[i]]

    Delaunay tetrahedral似乎不是根据三维体画出来的，而是三维表面画出来的。"""
    from mayavi import mlab
    if color is None:
        random.seed(3.14)
        color = (random.random(), random.random(), random.random())
    scene = mlab.triangular_mesh(tetrahedron[:, 0], tetrahedron[:, 1], tetrahedron[:, 2],
                                 [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
                                 color=color, opacity=opacity)
    return scene


def cuboid(cuboidVerticesX, cuboidVerticesY, cuboidVerticesZ, color=(1, 1, 0.5), opacity=1.0):
    """Draw a cuboid.

    Parameters:
    -----------
    cuboidVerticesX/Y/Z (np.ndarray, shape (2, 2, 2)): coordinates of the 8 vertices
        of a cuboid along X/Y/Z axis.
    """
    from mayavi import mlab
    scene = mlab.gcf()

    def plotPlane(slice1, slice2, slice3):
        """绘制长方体六个面中的某个面"""
        return mlab.triangular_mesh(cuboidVerticesX[slice1, slice2, slice3],
                                    cuboidVerticesY[slice1,
                                                    slice2, slice3],
                                    cuboidVerticesZ[slice1,
                                                    slice2, slice3],
                                    [(0, 1, 2), (1, 2, 3)], color=color, opacity=opacity)
    sliceAll = slice(None)
    plotPlane(sliceAll, sliceAll, 0)
    plotPlane(sliceAll, sliceAll, 1)
    plotPlane(sliceAll, 0, sliceAll)
    plotPlane(sliceAll, 1, sliceAll)
    plotPlane(0, sliceAll, sliceAll)
    plotPlane(1, sliceAll, sliceAll)
    return scene


class DisplayCube:

    @staticmethod
    def mpl(cube):
        import matplotlib.pyplot as plt

        verts, faces, *_ = marching_cubes(cube, 0)

        fig = plt.figure(figsize=(3.2, 3.2))
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces)
        # ax.set_aspect('equal')
        plt.axis('off')
        plt.show()
        return

    @staticmethod
    def vv(cube):
        import visvis as vv

        verts, faces, normals, values = marching_cubes(cube, 0)
        vv.mesh(np.fliplr(verts), faces, normals, values)
        vv.use().Run()
        return

    @staticmethod
    def mayavi(cube):
        from mayavi import mlab

        verts, faces, *_ = marching_cubes(cube, 0)
        mlab.options.offscreen = True
        mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces)
        mlab.show()
        return


class Violin:
    """Bad code..."""
    NpzFile = np.lib.npyio.NpzFile

    def __init__(self, dataSet: Union[NpzFile, Tuple[NpzFile]], name: Union[str, Tuple[str]]) -> None:
        dataSet = list(dataSet)
        name = list(name)
        for i in range(len(dataSet)):
            dataSet[i] = dict(dataSet[i].items())
            dataSet[i].pop('mask')
            dataSet[i] = pd.DataFrame(dataSet[i])
            dataSet[i].index = pd.MultiIndex.from_tuples(
                [(name[i], idx) for idx in dataSet[i].index])
        self.dataSet = pd.concat(dataSet, axis=0)
        self.name = name

    def plot(self, *, feature, setName, figsize=None, zhfontProperty=None, **kwargs):
        import seaborn as sns
        import matplotlib.pyplot as plt

        zhfont = {'font': zhfontProperty}
        axNum = len(setName)
        if axNum == 1:
            data = self.dataSet.loc[setName, feature]
            fig, ax = plt.subplots(figsize=figsize)
            sns.violinplot(data=data, ax=ax, **kwargs)
        elif axNum == 2:
            setName = list(setName)
            fig, ax = plt.subplots(figsize=figsize)
            data = self.dataSet.loc[setName, feature]
            js = self.jsHelper(data[setName[0]], data[setName[1]])
            data = pd.DataFrame(data)
            data['Data Set'] = None
            for name in setName:
                data.loc[name, 'Data Set'] = name
            sns.violinplot(data=data, x=[0]*len(data), y=feature, hue='Data Set',
                           split=True, ax=ax, **kwargs)
            ax.set_xticks([])
            ax.set_title(f"JS散度: {js}", fontdict=zhfont)
        elif axNum == 3:
            setName = list(setName)
            fig, ax = plt.subplots(1, 2, sharey=True,
                                   figsize=figsize, tight_layout=True)
            data = self.dataSet.loc[setName, feature]
            data = pd.DataFrame(data)
            data['Data Set'] = None
            for name in setName:
                data.loc[name, 'Data Set'] = name
            vaeData = data.loc[['Real', 'VAE']]
            ganData = data.loc[['Real', 'GAN']]
            sns.violinplot(data=vaeData, x=[0]*len(vaeData), y=feature, hue='Data Set',
                           split=True, ax=ax[0], **kwargs)
            sns.violinplot(data=ganData, x=[0]*len(ganData), y=feature, hue='Data Set',
                           split=True, ax=ax[1], **kwargs)
            jsVae = self.jsHelper(
                vaeData.loc['Real', feature].values, vaeData.loc['VAE', feature].values)
            jsGan = self.jsHelper(
                ganData.loc['Real', feature].values, ganData.loc['GAN', feature].values)
            ax[0].set_xticks([])
            ax[1].set_xticks([])
            ax[1].set_ylabel(None)
            ax[0].set_title(f"JS散度: {jsVae}", fontdict=zhfont)
            ax[1].set_title(f"JS散度: {jsGan}", fontdict=zhfont)
            # ax[1].set_ylabel(None)
        else:
            raise ValueError("Check the parameter `setName`!")
        return fig

    def jsHelper(self, vec1, vec2, nPoint=1001):
        """帮助计算同一个特征的两个分布之间的js散度。"""
        from scipy.stats import gaussian_kde
        from particle.utils.dirty import Entropy
        # 两个向量的极值
        extrema = np.array([np.sort(vec1)[[0, -1]],
                            np.sort(vec2)[[0, -1]]])
        # 设定特征（x）的变化范围
        xRange = np.linspace(extrema.min(), extrema.max(), nPoint)
        unitIntervalLength = (xRange[-1] - xRange[0]) / (nPoint - 1)
        # 计算概率密度density
        dsty1 = gaussian_kde(vec1).pdf(xRange)
        dsty2 = gaussian_kde(vec2).pdf(xRange)
        # 计算概率
        p1 = dsty1 * unitIntervalLength
        p2 = dsty2 * unitIntervalLength
        # 计算JS散度
        js = Entropy.JSDivergence(p1, p2)
        return round(js, 2)
