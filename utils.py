import re
import random
from xml.dom import minidom

import numpy as np
import scipy
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes_lewiner

import torch
from torch import nn
from mayavi import mlab


def sample_labels(size: int, lim: list, seed: int = None) -> np.ndarray:
    """What's meaning of sample_labels?"""
    if seed:
        np.random.seed(seed)
    all_sands = np.arange(*lim)  # lim是只含有两个int的列表[lo, hi]
    # 不放回简单随机抽样
    sampled_sands = np.sort(np.random.choice(all_sands, size, replace=False))
    return sampled_sands


def project(tensor: torch.tensor, dim: int):
    return torch.max(tensor, dim=dim).values


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


class DisplayCube:
    @staticmethod
    def plot_mpl(cube):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

        verts, faces, *_ = marching_cubes_lewiner(cube, 0)
        poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
        print(np.array(poly3d).shape)
        x, y, z = zip(*verts)

        fig = plt.figure(figsize=(3.2, 3.2))
        ax = plt.axes(projection='3d')
        # ax.set_aspect('equal')
        plt.axis('off')

        ax.scatter(x, y, z, alpha=0)
        ax.add_collection3d(Poly3DCollection(
            poly3d, facecolors='grey', linewidths=1, alpha=1))
        ax.add_collection3d(Line3DCollection(
            poly3d, colors='k', linewidths=0.3, linestyles='--', alpha=0.1))

        print(fig.canvas.get_width_height())
        plt.show()
        return

    @staticmethod
    def plot_vv(cube):
        import visvis as vv

        verts, faces, normals, values = marching_cubes_lewiner(cube, 0)
        vv.mesh(np.fliplr(verts), faces, normals, values)
        vv.use().Run()
        return

    @staticmethod
    def plot_mayavi(cube):
        from mayavi import mlab

        verts, faces, *_ = marching_cubes_lewiner(cube, 0)
        mlab.options.offscreen = True
        mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces)
        mlab.show()
        return


def make_grid(images: np.ndarray, filename: str, nrow: int = 8, normalize: bool = True):
    """Make a grid of images from input `images`.

    Parameters:
    -----------
    images: a batch of images, shape: (B, H, W, C)
    filename: the name of the image-grid file to be saved"""
    from torchvision.utils import save_image

    # get the batch, height, width, channel
    b, h, w, c = images.shape
    tensors = torch.empty((b, c, h, w), dtype=torch.float32)
    for i in range(b):
        for j in range(c):
            tensors[i, j] = torch.from_numpy(images[i, :, :, j])
    save_image(tensors, filename, nrow=nrow, normalize=normalize)
    return


class Circumsphere:
    """Copied from GitHub at https://github.com/shrx/mbsc.git, not my original.
    """
    @classmethod
    def fit(cls, array):
        """Compute exact minimum bounding sphere of a 3D point cloud (or a
        triangular surface mesh) using Welzl's algorithm.

        - X     : M-by-3 list of point co-ordinates or a triangular surface
                            mesh specified as a TriRep object.
        - R     : radius of the sphere.
        - C     : 1-by-3 vector specifying the centroid of the sphere.
        - Xb    : subset of X, listing K-by-3 list of point coordinates from
                            which R and C were computed. See function titled
                            'FitSphere2Points' for more info.

        REREFERENCES:
        [1] Welzl, E. (1991), 'Smallest enclosing disks (balls and ellipsoids)',
            Lecture Notes in Computer Science, Vol. 555, pp. 359-370

        Matlab code author: Anton Semechko (a.semechko@gmail.com)
        Date: Dec.2014"""

        # Get the convex hull of the point set
        hull = ConvexHull(array)
        hull_array = array[hull.vertices]
        hull_array = np.unique(hull_array, axis=0)
        # print(len(hull_array))

        # Randomly permute the point set
        hull_array = np.random.permutation(hull_array)

        if len(hull_array) <= 4:
            R, C = cls.fit_base(hull_array)
            return R, C, hull_array

        elif len(hull_array) < 1000:
            # try:
            R, C, _ = cls.B_min_sphere(hull_array, [])

            # Coordiantes of the points used to compute parameters of the
            # minimum bounding sphere
            D = np.sum(np.square(hull_array - C), axis=1)
            idx = np.argsort(D - R**2)
            D = D[idx]
            Xb = hull_array[idx[:5]]
            D = D[:5]
            Xb = Xb[D < 1E-6]
            idx = np.argsort(Xb[:, 0])
            Xb = Xb[idx]
            return R, C, Xb
            # except:
            #raise Exception
        else:
            M = len(hull_array)
            dM = min([M // 4, 300])
        # unnecessary ?
        #		res = M % dM
        #		n = np.ceil(M/dM)
        #		idx = dM * np.ones((1, n))
        #		if res > 0:
        #			idx[-1] = res
        #
        #		if res <= 0.25 * dM:
        #			idx[n-2] = idx[n-2] + idx[n-1]
        #			idx = idx[:-1]
        #			n -= 1

            hull_array = np.array_split(hull_array, dM)
            Xb = np.empty([0, 3])
            for i in range(len(hull_array)):
                R, C, Xi = cls.B_min_sphere(
                    np.vstack([Xb, hull_array[i]]), [])

                # 40 points closest to the sphere
                D = np.abs(np.sqrt(np.sum((Xi - C)**2, axis=1)) - R)
                idx = np.argsort(D, axis=0)
                Xb = Xi[idx[:40]]

            D = np.sort(D, axis=0)[:4]
            # print(Xb)
            # print(D)
            #print(np.where(D/R < 1e-3)[0])
            Xb = np.take(Xb, np.where(D/R < 1e-3)[0], axis=0)
            Xb = np.sort(Xb, axis=0)
            # print(Xb)

            return R, C, Xb

    @classmethod
    def fit_base(cls, array):
        """Fit a sphere to a set of 2, 3, or at most 4 points in 3D space. Note that
        point configurations with 3 collinear or 4 coplanar points do not have 
        well-defined solutions (i.e., they lie on spheres with inf radius).

        - X     : M-by-3 array of point coordinates, where M<=4.
        - R     : radius of the sphere. R=Inf when the sphere is undefined, as 
                    specified above.
        - C     : 1-by-3 vector specifying the centroid of the sphere. 
                    C=nan(1,3) when the sphere is undefined, as specified above.

        Matlab code author: Anton Semechko (a.semechko@gmail.com)
        Date: Dec.2014"""

        N = len(array)

        if N > 4:
            print('Input must a N-by-3 array of point coordinates, with N<=4')
            return

        # Empty set
        elif N == 0:
            R = np.nan
            C = np.full(3, np.nan)
            return R, C

        # A single point
        elif N == 1:
            R = 0.
            C = array[0]
            return R, C

        # Line segment
        elif N == 2:
            R = np.linalg.norm(array[1] - array[0]) / 2
            C = np.mean(array, axis=0)
            return R, C

        else:  # 3 or 4 points
            # Remove duplicate vertices, if there are any
            uniq, index = np.unique(array, axis=0, return_index=True)
            array_nd = uniq[index.argsort()]
            if not np.array_equal(array, array_nd):
                print("found duplicate")
                print(array_nd)
                R, C = cls.fit_base(array_nd)
                return R, C

            tol = 0.01  # collinearity/co-planarity threshold (in degrees)
            if N == 3:
                # Check for collinearity
                D12 = array[1] - array[0]
                D12 = D12 / np.linalg.norm(D12)
                D13 = array[2] - array[0]
                D13 = D13 / np.linalg.norm(D13)

                chk = np.clip(np.abs(np.dot(D12, D13)), 0., 1.)
                if np.arccos(chk)/np.pi*180 < tol:
                    R = np.inf
                    C = np.full(3, np.nan)
                    return R, C

                # Make plane formed by the points parallel with the xy-plane
                n = np.cross(D13, D12)
                n = n / np.linalg.norm(n)
                ##print("n", n)
                r = np.cross(n, np.array([0, 0, 1]))
                if np.linalg.norm(r) != 0:
                    # Euler rotation vector
                    r = np.arccos(n[2]) * r / np.linalg.norm(r)
                ##print("r", r)
                Rmat = scipy.linalg.expm(np.array([
                    [0., -r[2], r[1]],
                    [r[2], 0., -r[0]],
                    [-r[1], r[0], 0.]
                ]))
                ##print("Rmat", Rmat)
                #Xr = np.transpose(Rmat*np.transpose(array))
                Xr = np.transpose(np.dot(Rmat, np.transpose(array)))
                ##print("Xr", Xr)

                # Circle centroid
                x = Xr[:, :2]
                A = 2 * (x[1:] - np.full(2, x[0]))
                b = np.sum(
                    (np.square(x[1:]) - np.square(np.full(2, x[0]))), axis=1)
                C = np.transpose(np.linalg.solve(A, b))

                # Circle radius
                R = np.sqrt(np.sum(np.square(x[0] - C)))

                # Rotate centroid back into the original frame of reference
                C = np.append(C, [np.mean(Xr[:, 2])], axis=0)
                C = np.transpose(np.dot(np.transpose(Rmat), C))
                return R, C

            # If we got to this point then we have 4 unique, though possibly co-linear
            # or co-planar points.
            else:
                # Check if the the points are co-linear
                D12 = array[1] - array[0]
                D12 = D12 / np.linalg.norm(D12)
                D13 = array[2] - array[0]
                D13 = D13 / np.linalg.norm(D13)
                D14 = array[3] - array[0]
                D14 = D14 / np.linalg.norm(D14)

                chk1 = np.clip(np.abs(np.dot(D12, D13)), 0., 1.)
                chk2 = np.clip(np.abs(np.dot(D12, D14)), 0., 1.)
                if np.arccos(chk1)/np.pi*180 < tol or np.arccos(chk2)/np.pi*180 < tol:
                    R = np.inf
                    C = np.full(3, np.nan)
                    return R, C

                # Check if the the points are co-planar
                n1 = np.linalg.norm(np.cross(D12, D13))
                n2 = np.linalg.norm(np.cross(D12, D14))

                chk = np.clip(np.abs(np.dot(n1, n2)), 0., 1.)
                if np.arccos(chk)/np.pi*180 < tol:
                    R = np.inf
                    C = np.full(3, np.nan)
                    return R, C

                # Centroid of the sphere
                A = 2 * (array[1:] - np.full(len(array)-1, array[0]))
                b = np.sum(
                    (np.square(array[1:]) - np.square(np.full(len(array)-1, array[0]))), axis=1)
                C = np.transpose(np.linalg.solve(A, b))

                # Radius of the sphere
                R = np.sqrt(np.sum(np.square(array[0] - C), axis=0))

                return R, C

    @classmethod
    def B_min_sphere(cls, P, B):
        eps = 1E-6
        if len(B) == 4 or len(P) == 0:
            R, C = cls.fit_base(B)  # fit sphere to boundary points
            return R, C, P

        # Remove the last (i.e., end) point, p, from the list
        P_new = P[:-1].copy()
        p = P[-1].copy()

        # Check if p is on or inside the bounding sphere. If not, it must be
        # part of the new boundary.
        R, C, P_new = cls.B_min_sphere(P_new, B)
        if np.isnan(R) or np.isinf(R) or R < eps:
            chk = True
        else:
            chk = np.linalg.norm(p - C) > (R + eps)

        if chk:
            if len(B) == 0:
                B = np.array([p])
            else:
                B = np.array(np.insert(B, 0, p, axis=0))
            R, C, _ = cls.B_min_sphere(P_new, B)
            P = np.insert(P_new.copy(), 0, p, axis=0)
        return R, C, P


class Plotter:
    random.seed(3.14)

    @classmethod
    def randomColor(cls):
        color = (random.random(), random.random(), random.random())
        return color

    @classmethod
    def sphere(cls, center, radius, nPoints=100, opacity=1.0, color=None):
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
        color = cls.randomColor() if color is None else color
        scene = mlab.mesh(x, y, z, color=color, opacity=opacity)
        return scene

    @classmethod
    def tetrahedron(cls, tetrahedron, opacity=1.0, color=None):
        """Tetrahedron: tri.points[tri.simplices[i]]

        Delaunay tetrahedral似乎不是根据三维体画出来的，而是三维表面画出来的。"""
        color = cls.randomColor() if color is None else color
        scene = mlab.triangular_mesh(tetrahedron[:, 0], tetrahedron[:, 1], tetrahedron[:, 2],
                                     [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
                                     color=color, opacity=opacity)
        return scene

    @classmethod
    def cuboid(cls, cuboidVerticesX, cuboidVerticesY, cuboidVerticesZ, color=(1, 1, 0.5), opacity=1.0):
        """Draw a cuboid.

        Parameters:
        -----------
        cuboidVerticesX/Y/Z (np.ndarray, shape (2, 2, 2)): coordinates of the 8 vertices 
            of a cuboid along X/Y/Z axis.
        """
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


class Log:
    """HTML日志信息提取处理的相关操作"""

    @staticmethod
    def get_loss(log_file):
        """从保存的HTML日志文件中提取出loss的变化情况"""

        # 从语法角度讲，末尾究竟应不应该有\s来匹配每一行最后的隐形换行符?
        mode = re.compile(
            r'Loss_re: (\d+\.\d{4}), Loss_kl: (\d+\.\d{4}), Loss: (\d+\.\d{4})\n$')
        loss_re = []
        loss_kl = []
        loss = []
        with open(log_file, 'r') as log:
            for line in log:
                target = mode.search(line)
                if target:  # 确保有匹配到对象
                    loss_re.append(float(target.group(1)))
                    loss_kl.append(float(target.group(2)))
                    loss.append(float(target.group(3)))
        return loss_re, loss_kl, loss

    @staticmethod
    def get_time(log_file):
        """从保存的HTML日志文件中提取出训练时间"""

        # mode不能强制^，因为对应html中第一行以<pre>开头，使用^会忽略掉第一行的匹配
        mode = re.compile(r'Time cost so far: (\d+)h (\d+)min (\d+)s\n$')
        time = []
        with open(log_file) as log:
            for line in log:
                target = mode.search(line)
                if target:  # 确保有匹配到对象
                    h = target.group(1)
                    m = target.group(2)
                    s = target.group(3)
                    time.append(int(h)*60 + int(m) + int(s)/60)
        return time


def parseConfig(xmlFile):
    """Parse the xml configuration defining a structure of a neural network.

    Returns:
    --------
    nnParams(dict): note that, since Python 3.6, the default dict strcture 
        returned here is an ordered dict.
    """

    document = minidom.parse(xmlFile)
    nnLayer = document.documentElement
    nnParams = {}
    for layer in nnLayer.childNodes:
        if type(layer) is not minidom.Element:
            continue
        kwargs = {}
        for param in layer.childNodes:
            if type(param) is not minidom.Element:
                continue
            text = param.childNodes[0].data
            if text.isdigit():
                kwargs[param.tagName] = int(text)
            elif text == "true":
                kwargs[param.tagName] = True
            elif text == "false":
                kwargs[param.tagName] = False
            else:
                kwargs[param.tagName] = text
        nnParams[layer.tagName+"_"+layer.getAttribute("id")] = kwargs
    return nnParams


def constructOneLayer(layerType, layerParam):
    layer = nn.Sequential()
    if layerType.startswith("fc"):
        kwargs = {"in_features", "out_features", "bias"}
        kwargs = {key: layerParam[key] for key in kwargs}
        layer.add_module(layerType, nn.Linear(**kwargs))
        if layerParam["use_bn"]:
            layer.add_module("bn", nn.BatchNorm1d(kwargs["out_features"]))
    elif layerType.startswith("convTranspose"):
        ConvTranspose, BatchNorm = (nn.ConvTranspose2d, nn.BatchNorm2d) if "2d" in layerType \
            else (nn.ConvTranspose3d, nn.BatchNorm3d)
        kwargs = {"in_channels", "out_channels", "kernel_size",
                  "stride", "padding", "output_padding", "bias"}
        kwargs = {key: layerParam[key] for key in kwargs}
        layer.add_module(layerType, ConvTranspose(**kwargs))
        if layerParam["use_bn"]:
            layer.add_module("bn", BatchNorm(kwargs["out_channels"]))
    elif layerType.startswith("conv"):
        Conv, BatchNorm = (nn.Conv2d, nn.BatchNorm2d) if "2d" in layerType \
            else (nn.Conv3d, nn.BatchNorm3d)
        kwargs = {"in_channels", "out_channels",
                  "kernel_size", "stride", "padding", "bias"}
        kwargs = {key: layerParam[key] for key in kwargs}
        layer.add_module(layerType, Conv(**kwargs))
        if layerParam["use_bn"]:
            layer.add_module("bn", BatchNorm(kwargs["out_channels"]))
    else:
        raise Exception("xml configuration error!")

    # add the activation fucntion layer
    if layerParam["activate_mode"] == "relu":
        layer.add_module("activate", nn.ReLU())
    elif layerParam["activate_mode"] == "sigmoid":
        layer.add_module("activate", nn.Sigmoid())
    else:
        raise Exception(
            "activate_mode is error, expected to be 'relu' or 'sigmoid'")

    return layer


if __name__ == '__main__':
    file = '../log/process(32, 40, 1e-3, 200, 128).html'
    loss_re, loss_kl, loss = Log.get_loss(file)
    time = Log.get_time(file)
