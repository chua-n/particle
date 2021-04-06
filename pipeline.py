import os
from typing import Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull
import scipy.ndimage as ndi
from skimage import filters, exposure, measure, morphology, transform
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

from .utils.dirty import timer, Circumsphere


class Sand:

    """对单个沙土颗粒进行相关处理的类"""

    def __init__(self, cube: np.uint8):
        """cube是容纳单个颗粒的二值化的三维数字图像矩阵, verts为提取的沙土颗粒表面的顶点组成的点集；
        level的含义用于self.surface()函数
        Attributes:
        -----------
        cube: 64 × 64 × 64, np.uint8类型, 为“二值数组”，即其数值仅含0和255。
        verts: nPoints × 3"""

        self.cube = cube
        self._volume = None
        self._area = None
        self._verts = None
        self._faces = None
        self._convexHull = None

    def getBorder(self, connectivity=1, toCoords=False):
        """获得一个颗粒在cube中的边界元素，可返回图像形式或坐标形式。

        Parameters:
        -----------
        connectivity (int): 可以从1变化到3，3代表self.cube.ndim
        toCoords (bool): 选择是否返回坐标形式的边界元素

        Returns: 
        --------
        返回类型为np.array。
            1) 如果`toCoords=False`，数组的形状与self.cube相同，依然是图像格式；
            2) 如果`toCoords=True`，数组的形状为(n, 3)，每一行表示一个边界点的(x,y,z)坐标。
        """
        se = ndi.generate_binary_structure(3, connectivity)
        inner = morphology.binary_erosion(self.cube, selem=se)
        border = self.cube - inner
        if toCoords:
            property = measure.regionprops(border)[0]
            centroid = property.centroid
            coords = property.coords
            # 向坐标原点平移，以质心为原点
            coords = coords - centroid
            return coords
        else:
            return border

    def toCoords(self) -> np.ndarray:
        """将颗粒体素转化为在存储数组中的坐标的形式。

        Returns
        -------
        out: (nPoints, 3) ndarray
            Here 0 <= nPoints <= 64*64*64.
        """
        coords = np.nonzero(self.cube)
        coords = np.asarray(coords, dtype=np.int16)
        coords = coords.T
        return coords

    def findSurface(self, cube: np.ndarray = None, level: float = None):
        """原来的cube是三维数组表征的体素数据，此函数采用Lewiner-MC算法从cube中提取出沙粒表面。这个
        表面以一系列顶点verts和连接顶点的三角面faces组成，verts的形式为每个点的三维坐标组成的数组，
        faces也为二维数组，其每一行代表一个三角面，表示形式为组成这个三角面的三个点在verts中的3个索引。

        Parameters
        ----------
        volume: (M, N, P) array
            Input data volume to find isosurfaces（等值面）.
        level : float.
            Contour value to search for isosurfaces in `volume`. If not given or None,
            the average of the min and max of vol is used.

        Returns
        -------
        verts : (V, 3) array
            Spatial coordinates for V unique mesh vertices (至高点，顶点). Coordinate order
            matches input `volume` (M, N, P).
        faces : (F, 3) array
            Define triangular faces via referencing vertex indices from ``verts``.
            This algorithm specifically outputs triangles, so each face has
            exactly three indices.
        normals : (V, 3) array
            The normal direction (法线方向) at each vertex, as calculated from the
            data.
        values : (V, ) array
            Gives a measure for the maximum value of the data in the local region
            near each vertex. This can be used by visualization tools to apply
            a colormap to the mesh."""
        if cube is None:
            cube = self.cube
        # 只保留前两个参数
        self._verts, self._faces, *_ = measure.marching_cubes(cube, level,
                                                              method='lewiner')
        return self._verts, self._faces

    def visualize(self, cube=None, figure=None, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1),
                  vivid=True, voxel=False, **kwargs):
        """可视化颗粒的表面，可选择画实心的体素形式，或提取的三角网格的表面形式，后者可选择是否
        进行高斯模糊以逼真化展示。
        """
        from mayavi import mlab
        # foreground, background
        kwargs['figure'] = mlab.figure(figure=figure,
                                       fgcolor=fgcolor, bgcolor=bgcolor)
        if cube is None:
            cube = self.cube
        cube = cube.astype(np.float32)
        # 画体素时是否进行高斯模糊没有区别，因此体素图强制不进行高斯模糊
        if vivid and not voxel:
            cube = filters.gaussian(cube, sigma=0.8, mode='reflect')

        if voxel:
            return self._visualizeVoxel(cube, **kwargs)
        else:
            return self._visualizeTriMesh(cube, **kwargs)

    def _visualizeVoxel(self, cube, color=(0.65, 0.65, 0.65), glyph="cube", scale_mode="none", figure=None, **kwargs):
        from mayavi import mlab
        flatten = cube.reshape(-1)
        x, y, z = np.nonzero(cube)
        val = flatten[np.nonzero(flatten)]
        fig = mlab.points3d(x, y, z, val, color=color, mode=glyph,
                            figure=figure, scale_mode=scale_mode, **kwargs)
        return fig

    def _visualizeTriMesh(self, cube, color=(0.65, 0.65, 0.65), figure=None, **kwargs):
        from mayavi import mlab
        if "level" in kwargs:
            verts, faces = self.findSurface(cube, level=kwargs.pop("level"))
        else:
            verts, faces = self.findSurface(cube)
        figure = mlab.triangular_mesh(verts[:, 0],
                                      verts[:, 1],
                                      verts[:, 2],
                                      faces, color=color, figure=figure, **kwargs)
        return figure

    @staticmethod
    def savefig(fig_handle, filename, path=None, magnification='auto'):
        """保存某3D颗粒的图片到目的路径"""
        from mayavi import mlab
        if path:
            cwd = os.getcwd()
            os.chdir(path)
            mlab.savefig(filename, figure=fig_handle,
                         magnification=magnification)
            os.chdir(cwd)
        else:
            mlab.savefig(filename, figure=fig_handle,
                         magnification=magnification)
        return

    def rotateInAPlane(self, angle, axes=(1, 0), cube=None):
        """平行于某平面对三维颗粒进行旋转。

        Parameters:
        -----------
        angle (float): The rotation angle in degrees.
        axes (tuple of two ints): The two axes that define the plane of rotation.

        Note:
        -----
            1. `ndi.rotate()`函数旋转完之后的数组其数值不再是二值化的，而是0~255之间的数都可能有；
            2. 由上，这里以中值为界，仍然将数组二值化，而之所以采用中值为阈值，是参考了`marching_cubes()`的`level`参数；
            3. 上面这种二值化的方式以及是否真的需要进行二值化实在还是有待商榷。
        """
        if cube is None:
            cube = self.cube
        rotated = ndi.rotate(cube, angle, axes, reshape=False)
        mid = (rotated.max() - rotated.min()) / 2
        zerosMask = rotated < mid
        rotated[zerosMask] = 0
        rotated[~zerosMask] = 255
        return rotated

    def randomRotate(self, cube=None):
        """随机旋转颗粒。"""
        rotated = self.cube if cube is None else cube
        angles = 360 * np.random.rand(3)
        rotated = self.rotateInAPlane(angles[0], (0, 1), rotated)
        rotated = self.rotateInAPlane(angles[1], (0, 2), rotated)
        rotated = self.rotateInAPlane(angles[2], (1, 2), rotated)
        return rotated

    def pca(self, method=None):
        """通过主成分分析，得到颗粒三个主成分的方向及长度，即颗粒处于相互正交的三个不同方向的最长长度。

        Note:
        -----
            主成分分析法以方差变化最大的方向作为最长方向，然而方差最大是否真的“变化范围大”值得进一步分析，
        这在本文中似乎并不一定完全准确。
        """
        if method is None:
            # np.con()返回一个array_like各行之间协方差(convariance)的矩阵
            covarMatrix = np.cov(self.toCoords().T)
            # np.linalg.eig(square_array) 返回输入方阵的特征值和右特征向量
            eigenvalues, eigenvectors = np.linalg.eig(covarMatrix)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            # eigenvectors中以每列为特征向量，这里将其转置为以每行为特征向量
            eigenvectors = eigenvectors[:, order].T
            return eigenvalues, eigenvectors
        elif method == "sklearn":
            from sklearn.decomposition import PCA
            pcaModel = PCA(n_components=3).fit(self.toCoords())
            pc = pcaModel.components_  # 主成分
            var = pcaModel.explained_variance_  # 方差
            return pc, var
        else:
            raise ValueError("Parameter `method` must be None or 'sklearn'!")

    def poseNormalization(self):  # 这部分很有问题啊！！
        """位姿归一化。这个函数的实现原理尚不明确！"""
        _, eigenvectors = self.pca()
        inv = np.linalg.inv(eigenvectors[::-1])  # 求逆矩阵
        offset = np.ones(3) * (self.cube.shape[0] / 2.0 - 0.5)
        offset_1 = offset - np.dot(inv, offset)
        output = ndi.affine_transform(
            self.cube, inv, offset_1, self.cube.shape, np.float32)
        output = np.where(output >= 0.5, 1, 0).astype(np.uint8)
        index = np.where(output == 1)
        npoints_of_index = len(index[0])
        temp = np.eye(3, 3)
        for i in range(3):
            if np.sum(index[i]) < 2 * npoints_of_index * offset[i]:
                temp[i, i] = -1
        inv = inv * temp
        offset_2 = offset - np.dot(inv, offset)
        output_nani = ndi.affine_transform(
            self.cube, inv, offset_2, self.cube.shape, np.float32)
        output_nani = np.where(output >= 0.5, 1, 0).astype(np.uint8)
        return output_nani

    def resize(self, newSize: Union[int, Tuple[int]], thrd: float = None) -> np.ndarray:
        """将颗粒cube缩放到新设置的尺寸。

        Parameters:
        -----------
        newSize(int or tuple of ints): new Cube size
        thrd(float): 默认缩放完后的cube为0~1之间的小数，可设定阈值将其转换为二值数组，
            同`rotateInAPlane()`，一般应设置为0.5。
        """
        if not hasattr(newSize, "__iter__"):  # if newSize is not a sequence
            newSize = (newSize,)*3
        newCube = transform.resize(self.cube, newSize)
        if thrd is not None:
            zerosMask = newCube < thrd
            newCube[zerosMask] = 0
            newCube[zerosMask] = 255
            newCube = newCube.astype(np.uint8)
        return newCube

    def rescale(self, ratio: Union[float, Tuple[float]], thrd: float = None) -> np.ndarray:
        """将颗粒cube按比例缩放。

        Parameters:
        -----------
        ratio(float or tuple of floats): scale factors
        thrd(float): 同`resize()`
        """
        if not hasattr(ratio, "__iter__"):
            ratio = (ratio,)*3
        newCube = transform.rescale(self.cube, ratio)
        if thrd is not None:
            zerosMask = newCube < thrd
            newCube[zerosMask] = 0
            newCube[zerosMask] = 255
            newCube = newCube.astype(np.uint8)
        return newCube

    # ----------------------------以下为计算颗粒的几何特征参数--------------------------

    def sandConvexHull(self, level: float = None) -> ConvexHull:
        """获取沙颗粒的凸包
        ConvexHull is a class that calculates the convex hull（凸包） of a given point set.
        |  Parameters
        |  ----------
        |  points : ndarray of floats, shape (npoints, ndim)
        |      Coordinates of points to construct a convex hull from
        |
        |  Attributes
        |  ----------
        |  points : ndarray of double, shape (npoints, ndim)
        |      Coordinates of input points.
        |  vertices : ndarray of ints, shape (nvertices,)
        |      Indices of points forming the vertices of the convex hull.
        |      For 2-D convex hulls, the vertices are in counterclockwise order.
        |      For other dimensions, they are in input order.
        |  simplices : ndarray of ints, shape (nfacet, ndim)
        |      Indices of points forming the simplical facets of the convex hull.
        |  neighbors : ndarray of ints, shape (nfacet, ndim)
        |      Indices of neighbor facets for each facet.
        |      The kth neighbor is opposite to the kth vertex.
        |      -1 denotes no neighbor.
        |  area : float
        |      Area of the convex hull.
        |
        |      .. versionadded:: 0.17.0
        |  volume : float
        |      Volume of the convex hull."""
        if self._verts is None:
            self.findSurface(level=level)
        convex_hull = ConvexHull(self._verts)
        self._convexHull = convex_hull
        return convex_hull

    def circumscribedSphere(self, level: float = None):
        """返回沙土颗粒的最小外接球的半径和球心坐标，似乎也是调用的人家的包"""
        if self._verts is None:
            self.findSurface(level=level)
        radius, centre, *_ = Circumsphere.fit(self._verts)
        return radius, centre

    def equEllipsoidalParams(self) -> Tuple[float, float, float]:
        """返回颗粒等效椭球的长、中、短轴的长度，half_axis, long >> small"""
        long, medium, short = 2 * np.sqrt(self.pca()[0])
        return long, medium, short

    def surfaceArea(self) -> float:
        """计算沙土颗粒的表面积"""
        verts, faces = self.findSurface(level=0)
        area = measure.mesh_surface_area(verts, faces)
        self._area = area
        return area

    def volume(self) -> float:
        volume = np.sum(self.cube == 1)
        self._volume = volume
        return volume

    def sphericity(self) -> float:
        """计算球度sphericity"""
        volume = self.volume() if self._volume is None else self._volume
        area = self.surfaceArea() if self._area is None else self._area
        return (36*np.pi*volume**2)**(1/3) / area

    def EI_FI(self) -> tuple:
        """计算伸长率EI和扁平率FI"""
        l, m, s = self.equEllipsoidalParams()
        return l/m, m/s

    def convexity(self) -> float:
        """计算凸度：颗粒体积与凸包体积之比"""
        convex_hull = self.sandConvexHull() if self._convexHull is None else self._convexHull
        volume = self.volume() if self._volume is None else self._volume
        return volume / convex_hull.volume

    def angularity(self) -> float:
        """计算颗粒棱角度(angularity).定义为凸包表面积 P_c 和等效椭球表面积 P_e 之比。"""
        a, b, c = self.equEllipsoidalParams()
        P_e = 4*np.pi*(((a*b)**1.6+(a*c)**1.6+(b*c)**1.6)/3)**(1/1.6)
        convex_hull = self.sandConvexHull() if self._convexHull is None else self._convexHull
        P_c = convex_hull.area
        return P_c / P_e

    def roughness(self) -> float:
        """计算颗粒的粗糙度。"""
        surf_p = self.surfaceArea()
        convex_hull = self.sandConvexHull() if self._convexHull is None else self._convexHull
        surf_c = convex_hull.area
        return surf_p/surf_c


class SandHeap:
    """对整个沙土体的处理流程——从CT扫描图到单颗粒提取。
    """

    def __init__(self, source: str = "/media/chuan/000935950005A663/liutao/ct-images/",
                 se: np.array = ndi.generate_binary_structure(rank=3, connectivity=2),
                 connectivity: int = 1,
                 ratio: float = 1,
                 persistencePath: str = "./data/liutao/",
                 cubeSize: int = 64
                 ):
        """
        Parameters:
        -----------
        source: path of input source data
        se: structure element for morphological openrations
        connectivity: pixel connectivity for watershed segment and particle extraction.
            Default 1 is because its accuracy in practice, `connectivity=2` performs 
            unexpectedly bad.
        persistencePath: path to save calculated important data permanently
        cubeSize: size of the cube to contain a soil particle

        Attributes:
        -----------
        """
        self.status = None
        self._loadData(source)  # get `self.data`
        self._getCircleMask(ratio)  # get `self.circleMask`
        self.se = se
        self.connectivity = connectivity
        self.persistencePath = persistencePath
        self.cubeSize = cubeSize
        self._distance = None
        self._markers = None
        self.preSegmented = None
        self.finalSegmented = None

    def setStatus(self, name):
        assert name in {"data-loaded", "histogram-equalized", "filtered",
                        "contrast-enhanced", "binary-segmented", "binary-opened",
                        "holes-filled", "distance-calculated", "markers-calculated",
                        "pre-segmented", "final-segmented"}
        self.status = name

    def checkStatus(name):
        from functools import wraps

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self = args[0]  # 先用这种奇葩的用法，后续再了解怎么获取self以及应用在非实例方法
                if isinstance(name, str) and self.status != name:
                    assert False, f"Action Error! The function `{func.__name__}` shouldn't " \
                        f"be executed when status is not {name}, make the status correct!\n" \
                        f"`{func.__name__}` has exited without execution!"
                elif isinstance(name, list) and self.status not in name:
                    assert False, f"Action Error! The function `{func.__name__}` shouldn't "\
                        f"be executed when status does not belong to {name}, make the status correct!\n" \
                        f"`{func.__name__}` has exited without execution!"
                else:
                    res = func(*args, **kwargs)
                    return res
            return wrapper
        return decorator

    @timer
    def _loadData(self, source: str):
        assert source is None or os.path.isdir(
            source) or source.endswith(".npy")
        if source is None:
            self.data = None
            return
        elif os.path.isdir(source):
            from skimage import io
            cwd = os.getcwd()
            os.chdir(source)
            files = os.listdir()
            files.sort(key=lambda filename: int(filename.split('.')[0]))
            sample = io.imread(files[0])
            self.data = np.empty(
                (len(files), *sample.shape), dtype=sample.dtype)
            for i, file in enumerate(files):
                self.data[i] = io.imread(file)
            os.chdir(cwd)
        else:
            self.data = np.load(source)
        self.setStatus("data-loaded")

    def _getCircleMask(self, ratio=1):
        if self.data is None:
            self.circleMask = None
            return
        from skimage import draw
        sample = self.data[0]
        h, w = sample.shape
        assert h == w
        size = h
        diameter = size * ratio
        circleMask = np.zeros_like(sample, dtype=bool)
        rr, cc = draw.disk((size/2, size/2), diameter/2, shape=sample.shape)
        circleMask[rr, cc] = True
        pln = self.data.shape[0]  # 扩展出pln维度
        circleMask = np.expand_dims(circleMask, 0).repeat(pln, axis=0)
        self.circleMask = circleMask

    def drawHistogram(self, nbins=256):
        print("Now plotting the histogram...")
        n, bins, patches = plt.hist(self.data[self.circleMask], bins=nbins)
        print("Plotting Complete!")
        return n, bins, patches

    @timer
    @checkStatus("data-loaded")
    def equalizeHist(self, nbins=256, draw=False):
        res = exposure.equalize_hist(
            self.data, nbins=nbins, mask=self.circleMask)
        self.data = img_as_ubyte(res)
        self.setStatus("histogram-equalized")
        if draw:
            self.drawHistogram(nbins=nbins)

    @timer
    @checkStatus(["data-loaded", "histogram-equalized"])
    def filter(self, mode="median", cycle=3, package="ndimage", draw=False, persistence=True):
        assert mode in ("median", "mean") and \
            package in ("ndimage", "skimage")
        for _ in range(cycle):
            if package == "ndimage":
                if mode == "median":
                    ndi.median_filter(self.data, size=3,
                                      mode="constant", cval=0, output=self.data)
                else:
                    ndi.uniform_filter(self.data, size=3,
                                       mode="constant", cval=0, output=self.data)
            else:
                if mode == "median":
                    filters.rank.median(self.data, selem=self.se,
                                        mask=self.circleMask, out=self.data)
                else:
                    filters.rank.mean(self.data, selem=self.se,
                                      mask=self.circleMask, out=self.data)
        self.setStatus("filtered")
        if persistence:
            np.save(os.path.join(self.persistencePath,
                                 self.status+".npy"), self.data)
        if draw:
            self.drawHistogram()

    @timer
    @checkStatus("filtered")
    def enhanceContrast(self, draw=False, persistence=True):
        filters.rank.enhance_contrast(
            self.data, selem=self.se, out=self.data, mask=self.circleMask)

        self.setStatus("contrast-enhanced")
        if persistence:
            np.save(os.path.join(self.persistencePath,
                                 self.status+".npy"), self.data)
        if draw:
            self.drawHistogram()

    @timer
    @checkStatus(["contrast-enhanced", "filtered"])
    def binarySegmentation(self, threshold=108):
        mask = self.data > threshold
        self.data = self.data.astype(bool)
        self.data[mask] = True
        self.data[~mask] = False
        self.setStatus("binary-segmented")
        return

    @timer
    @checkStatus("binary-segmented")
    def binaryOpening(self, iters=1):
        ndi.binary_opening(self.data, structure=self.se,
                           iterations=iters, mask=self.circleMask, output=self.data)
        self.setStatus("binary-opened")
        return

    @timer
    @checkStatus("binary-opened")
    def binaryFillHoles(self):
        ndi.binary_fill_holes(self.data, structure=self.se, output=self.data)
        self.setStatus("holes-filled")
        return

    @timer
    @checkStatus("holes-filled")
    def _distanceForWatershed(self, mode="cdt", metric="chessboard", pinch=True, persistence=True):
        self.circleMask = None
        if mode == "cdt":
            self._distance = ndi.distance_transform_cdt(
                self.data, metric=metric)
        elif mode == "edt":
            self._distance = ndi.distance_transform_edt(self.data)
        else:
            raise ValueError(
                "Parameter `mode` should be either 'cdt' or 'edt'!")
        # 已经证明distance使用np.float64或np.float32对后续的计算丝毫不影响
        # 所以为了节省空间、提高计算效率，还是开启pinch选项使用float32吧
        if pinch and mode == "edt":  # 压缩内存占用
            self._distance = self._distance.astype(np.float32)
        if persistence:
            np.save(os.path.join(self.persistencePath,
                                 "distance.npy"), self._distance)
        self.setStatus("distance-calculated")

    @timer
    @checkStatus("distance-calculated")
    def _markersForWatershed(self, min_distance=7, pinch=True, persistence=True):
        self.circleMask = None
        distance = self._distance
        coords = peak_local_max(distance, min_distance=min_distance,
                                labels=self.data)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        self._markers = measure.label(mask, connectivity=self.connectivity)
        if pinch:  # 压缩内存占用
            self._markers = self._markers.astype(np.int32)
        if persistence:
            np.save(os.path.join(self.persistencePath,
                                 "markers.npy"), self._markers)
        self.setStatus("markers-calculated")

    @timer
    @checkStatus(["holes-filled", "distance-calculated", "markers-calculated"])
    def watershedSegmentation(self, min_distance=7, persistence=True):
        """注：这里的分水岭分割内部使用的其实是“1级连通”，正好符合了我的主动设定`self.connectivity=1`。
        """
        # assert method in ("distance", "gradient")
        self.circleMask = None
        if self._distance is None:
            self._distanceForWatershed()
        if self._markers is None:
            self._markersForWatershed(min_distance)

        segmented = watershed(-self._distance, self._markers,
                              mask=self.data, watershed_line=True)
        print(
            f"After preliminary watershed-segmentation, found {segmented.max()} regions.")
        segmented = segmented.astype(bool)  # 选择节省空间否？
        self._distance = None
        self._markers = None
        self.preSegmented = segmented
        self.setStatus("pre-segmented")
        if persistence:
            np.save(os.path.join(self.persistencePath,
                                 self.status+".npy"), segmented)

    @timer
    @checkStatus("pre-segmented")
    def removeBigSegmentationFace(self, mode="3d", threshold=300, connectivity=None, persistence=True, returnDiagram=False):
        self.circleMask = None
        segmentationFace = img_as_ubyte(
            self.data) - img_as_ubyte(self.preSegmented)
        if mode == "3d":
            labeledFace, num = measure.label(segmentationFace, return_num=True,
                                             connectivity=2 if connectivity is None else connectivity)
            print(
                f"The pre-segmented image has initially {num} segmentation faces.")
            regions = measure.regionprops(labeledFace)
            inds = [i for i, region in enumerate(regions)
                    if region.area >= threshold]
            print(f"{len(inds)} segmentation faces will be removed!")
            for i in inds:
                coords = tuple(regions[i].coords.T)
                segmentationFace[coords] = 0
            print("Removal completes!")
        elif mode == "2d":
            from tqdm import tqdm
            initNum = removalNum = 0
            for i, crossSection in tqdm(enumerate(segmentationFace)):
                labeledLine, n = measure.label(
                    crossSection, return_num=True, connectivity=2)
                initNum += n
                regions = measure.regionprops(labeledLine)
                inds = [i for i, region in enumerate(regions)
                        if region.area >= threshold]
                removalNum += len(inds)
                for ind in inds:
                    coords = tuple(regions[ind].coords.T)
                    segmentationFace[i][coords] = 0
            print(
                f"The pre-segmented image has initially {initNum} segmentation lines totally.")
            print(f"{removalNum} segmentation lines have been removed!")
        else:
            raise ValueError("Parameter `mode` should be either '2D' or '3D'.")

        segmented = self.data + segmentationFace
        self.finalSegmented = segmented
        self.setStatus("final-segmented")
        if persistence:
            np.save(os.path.join(self.persistencePath,
                                 self.status+".npy"), segmented)
        if returnDiagram:
            areas = [region.area for region in regions]
            areasAfterRemoval = [region.area for region in regions
                                 if region.area < threshold]
            fig, ax = plt.subplots(1, 2, figsize=(12, 9))
            ax[0].violinplot(areas, showextrema=False)
            ax[0].set(title="Total Segmentation Faces", ylabel="voxels")
            ax[1].violinplot(areasAfterRemoval, showextrema=False)
            ax[1].set(title="Segmentation Faces After Removal", ylabel="voxels")
            return fig

    @staticmethod
    @timer  # @classmethod与@staticmethod装饰器必须置于顶层
    def removeBoundaryLabels(labeled, inplace=True):
        img = labeled
        boundaryLabels = set()
        index = [slice(None) for _ in range(img.ndim)]
        for dim in range(img.ndim):
            for boundarySlice in [slice(1), slice(-1, None)]:
                index[dim] = boundarySlice
                for val in np.unique(img[tuple(index)].ravel()):
                    if val != 0:
                        boundaryLabels.add(val)
            index[dim] = slice(None)
        res = img if inplace else img.copy()
        for label in boundaryLabels:
            mask = res != label
            res[:] *= mask
            # 比下面的写法更鲁棒些，因为不知道背景元素的值，可能不是0
            # mask = res == label
            # res[mask] = 0
        return None if inplace else res

    @timer
    @checkStatus("final-segmented")
    def putIntoCube(self) -> np.ndarray:
        self.circleMask = None
        """将每一个颗粒分别提取到cube里。"""
        labeledImage, num = measure.label(
            self.finalSegmented, return_num=True, connectivity=self.connectivity)
        cubes = np.zeros((num, self.cubeSize, self.cubeSize, self.cubeSize),
                         dtype=bool)
        regions = measure.regionprops(labeledImage)
        for i, region in enumerate(regions):
            particle = region.image
            particleShape = np.array(particle.shape)
            initialIndex = (self.cubeSize - particleShape) // 2
            cubes[i][initialIndex[0]:initialIndex[0]+particle.shape[0],
                     initialIndex[1]:initialIndex[1]+particle.shape[1],
                     initialIndex[2]:initialIndex[2]+particle.shape[2]] = particle
        return cubes
