import os
import numpy as np

import scipy.ndimage as ndi
from skimage import filters, exposure, measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage.util import img_as_ubyte

from .utils.dirty import sample_labels, Circumsphere, timer


class Sand:

    """对单个沙土颗粒进行相关处理的类"""

    def __init__(self, cube: np.uint8):
        """cube是容纳单个颗粒的二值化的三维数字图像矩阵, verts为提取的沙土颗粒表面的顶点组成的点集；
        level的含义用于self.surface()函数
        Attributes:
        -----------
        cube: 64 × 64 × 64, np.uint8类型, 最大值为1而非255
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

    def point_cloud(self) -> np.ndarray:
        """转化为点云，尚不知有何用
        Returns
        -------
            out: (3, nPoints) ndarray
                Here 0 <= nPoints <= 64*64*64."""

        cloud = np.asarray(np.where(self.cube != 0), dtype=np.int32)
        return cloud

    def surface(self, cube: np.ndarray = None, level: float = None):
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
        verts, faces = self.surface(cube)
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

    def rotate(self, angle, axes=(1, 0)):  # 这个axes和下面的random_rotate是联系的，要搞懂
        """旋转颗粒"""
        # 如果cube里面出现了0和1以外的值就报错（应该是威哥犯过这样的错误），这行代码挺耗时间的
        assert len(np.unique(self.cube)) == 2
        rotated = ndi.rotate(self.cube, angle, axes, reshape=False)
        # 这样分域值的理由是什么？？？
        return np.where(rotated >= 0.5, 1, 0).astype(np.uint8)

    def random_rotate(self):
        """随机旋转颗粒"""
        angle = 360 * np.random.rand()
        axes = sample_labels(2, [0, 3])  # 终于用上你了，不会就这点用吧？
        return self.rotate(angle, axes)

    def pca_eig(self, cloud):
        """应该是pca_eigenvalue，这里的主成分代表什么意思？？？"""
        # np.con()返回一个array_like的各行之间的协方差的矩阵(convariance)
        inertia = np.cov(cloud)  # inertia：[力]惯性；惰性，迟钝
        # np.linalg.eig(square_array) 返回输入方阵的特征值和右特征向量
        e_values, e_vectors = np.linalg.eig(inertia)
        # np.argsort() returns the indices that would sort an array.
        order = np.argsort(e_values)
        eval3, eval2, eval1 = e_values[order]
        axis3, axis2, axis1 = e_vectors[:, order].transpose()
        return [eval3, eval2, eval1], [axis3, axis2, axis1]

    def pose_normalization(self):  # 这部分很有问题啊！！
        """位姿归一化"""
        cloud = self.point_cloud()
        _, e_vectors = self.pca_eig(cloud)
        inv = np.linalg.inv(e_vectors)  # 求逆矩阵
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

    def rescale(self, new_size: int):
        """将颗粒cube缩放到设置的尺寸new_size"""
        ratio = new_size / self.cube.shape[0]
        rescaled = ndi.zoom(self.cube, ratio, output=np.float32)
        rescaled = np.where(rescaled >= 0.5, 1, 0).astype(np.uint8)
        return rescaled

    def get_zernike_moments(self, order=6, scale_input=True, decimate_fraction=0,
                            decimate_smooth=25, verbose=False):
        """计算沙土颗粒的Zernike矩, moment是矩的意思。调得人家的包，没什么好说的。"""
        from particle.mindboggle.shapes.zernike.zernike import zernike_moments
        if self._verts is None or self._faces is None:
            self.surface(self.cube)
        descriptors = zernike_moments(self._verts, self._faces, order, scale_input,
                                      decimate_fraction, decimate_smooth,
                                      verbose)
        return descriptors

    # ----------------------------以下为计算颗粒的几何特征参数--------------------------

    def sand_convex_hull(self, level: float = None) -> ConvexHull:
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
            self.surface(level=level)
        convex_hull = ConvexHull(self._verts)
        self._convexHull = convex_hull
        return convex_hull

    def circumscribed_sphere(self, level: float = None):
        """返回沙土颗粒的最小外接球的半径和球心坐标，似乎也是调用的人家的包"""
        if self._verts is None:
            self.surface(level=level)
        radius, centre, *_ = Circumsphere.fit(self._verts)
        return radius, centre

    def equ_ellipsoidal_params(self):
        """返回颗粒等效椭球的长、中、短轴的长度，half_axis, long >> small"""
        cloud = self.point_cloud()
        a, b, c = 2 * np.sqrt(self.pca_eig(cloud)[0])
        return c, b, a

    def surf_area(self) -> float:
        """计算沙土颗粒的表面积"""
        verts, faces = self.surface(level=0)
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
        area = self.surf_area() if self._area is None else self._area
        return (36*np.pi*volume**2)**(1/3) / area

    def EI_FI(self) -> tuple:
        """计算伸长率EI和扁平率FI"""
        l, m, s = self.equ_ellipsoidal_params()
        return l/m, m/s

    def convexity(self) -> float:
        """计算凸度：颗粒体积与凸包体积之比"""
        convex_hull = self.sand_convex_hull() if self._convexHull is None else self._convexHull
        volume = self.volume() if self._volume is None else self._volume
        return volume / convex_hull.volume

    def angularity(self) -> float:
        """计算颗粒棱角度(angularity).定义为凸包表面积 P_c 和等效椭球表面积 P_e 之比。"""
        a, b, c = self.equ_ellipsoidal_params()
        P_e = 4*np.pi*(((a*b)**1.6+(a*c)**1.6+(b*c)**1.6)/3)**(1/1.6)
        convex_hull = self.sand_convex_hull() if self._convexHull is None else self._convexHull
        P_c = convex_hull.area
        return P_c / P_e

    def roughness(self) -> float:
        """计算颗粒的粗糙度。"""
        surf_p = self.surf_area()
        convex_hull = self.sand_convex_hull() if self._convexHull is None else self._convexHull
        surf_c = convex_hull.area
        return surf_p/surf_c

    def cal_geo_feat_cube(self) -> list:  # dict更好
        pass


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
