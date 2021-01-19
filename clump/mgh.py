"""Reproduction of the algorithm presented by paper "Multi-sphere approximation of real particles for \
DEM simulation based on a modified greedy heuristic algorithm (https://doi.org/10.1016/j.powtec.2015.08.026)". 

@author: chuan
"""


import numpy as np
from sklearn.decomposition import PCA
from mayavi import mlab
from particle.pipeline import Sand
import particle.utils.plotter as plotter


class CellCollection:
    """Just a data structure of the cell collection.
    """

    def __init__(self, sandPointCloud, nX=40):
        self.nX = nX
        pcaReturns = CellCollection._pcaTransfer(sandPointCloud)
        self.pcaModel = pcaReturns[0]
        self.pcaPoints = pcaReturns[1]
        self.l, self.w, self.h = pcaReturns[2]
        self.lEnds, self.wEnds, self.hEnds = pcaReturns[3]
        self.r = self.l / self.nX / 2
        self.nY = int(self.w / self.l * self.nX)  # 四舍五入还是地板除法，需要搞清楚分割方法
        self.nZ = int(self.h / self.l * self.nX)
        # cellCoords是每一个cell中心点的坐标
        self.cellBox, self.cellInd, self.cellCoords = self._getCells()
        self.boundaryCoords, self.innerCoords = self._boundaryDetect()

    @classmethod
    def _pcaTransfer(cls, sandPointCloud):
        pcaModel = PCA(n_components=3)
        pcaModel = pcaModel.fit(sandPointCloud)
        pcaPoints = pcaModel.transform(sandPointCloud)
        l = pcaPoints[:, 0].max() - pcaPoints[:, 0].min()
        w = pcaPoints[:, 1].max() - pcaPoints[:, 1].min()
        h = pcaPoints[:, 2].max() - pcaPoints[:, 2].min()
        assert l >= w >= h
        # [l, w, h]Ends denotes the two ends of the line along l, w, h axis
        lEnds = (pcaPoints[:, 0].min(), pcaPoints[:, 0].max())
        wEnds = (pcaPoints[:, 1].min(), pcaPoints[:, 1].max())
        hEnds = (pcaPoints[:, 2].min(), pcaPoints[:, 2].max())
        return pcaModel, pcaPoints, [l, w, h], [lEnds, wEnds, hEnds]

    def _getCuboid(self):
        cuboidVerticesL, cuboidVerticesW, cuboidVerticesH = np.meshgrid(
            self.lEnds, self.wEnds, self.hEnds)
        return cuboidVerticesL, cuboidVerticesW, cuboidVerticesH

    def _getCells(self):
        # 重新命名方便后续书写
        nX, nY, nZ = self.nX, self.nY, self.nZ
        r = self.r
        lEnds, wEnds, hEnds = self.lEnds, self.wEnds, self.hEnds
        pcaPoints = self.pcaPoints

        # 对cellBox三维数组的索引，记坐标(lEnds[0], wEnds[0], hEnds[0])为索引(0, 0, 0)
        # clct: collection, ind: index, ctr: center
        cellBox = np.zeros((nX, nY, nZ), dtype=np.uint8)
        cellInd = np.empty_like(pcaPoints, dtype=np.uint)
        cellInd[:, 0] = (pcaPoints[:, 0] - lEnds[0]) // (2 * r)
        cellInd[:, 1] = (pcaPoints[:, 1] - wEnds[0]) // (2 * r)
        cellInd[:, 2] = (pcaPoints[:, 2] - hEnds[0]) // (2 * r)
        # 这个oriented bounding box是一个闭区间，而上式计算的时候是左闭右开区间
        # 因而需要把左闭右开区间计算出来的最大端方向的索引减一，归结到其小端的cell上
        cellInd[cellInd[:, 0] == nX, 0] = nX - 1
        cellInd[cellInd[:, 1] == nY, 1] = nY - 1
        cellInd[cellInd[:, 2] == nZ, 2] = nZ - 1
        cellBox[cellInd[:, 0],
                cellInd[:, 1],
                cellInd[:, 2]] = 1

        # 变相地索引去重
        cellInd = np.vstack(np.nonzero(cellBox == 1)).T
        cellCoords = np.empty_like(cellInd, dtype=float)
        cellCoords[:, 0] = lEnds[0] + r + cellInd[:, 0] * 2 * r
        cellCoords[:, 1] = wEnds[0] + r + cellInd[:, 1] * 2 * r
        cellCoords[:, 2] = hEnds[0] + r + cellInd[:, 2] * 2 * r
        return cellBox, cellInd, cellCoords

    def _boundaryDetect(self):
        isBoundary = np.ones(len(self.cellInd), dtype=bool)
        boxShape = self.cellBox.shape
        for i, (indL, indW, indH) in enumerate(self.cellInd):
            if (0 < indL < boxShape[0] - 1 and
                0 < indW < boxShape[1] - 1 and
                0 < indH < boxShape[2] - 1 and
                self.cellBox[indL - 1, indW, indH] == 1 and
                self.cellBox[indL + 1, indW, indH] == 1 and
                self.cellBox[indL, indW - 1, indH] == 1 and
                self.cellBox[indL, indW + 1, indH] == 1 and
                self.cellBox[indL, indW, indH - 1] == 1 and
                self.cellBox[indL, indW, indH + 1] == 1
                ):
                isBoundary[i] = False

        # boundaryInd = self.cellInd[isBoundary]
        boundaryCoords = self.cellCoords[isBoundary]

        # innerInd = self.cellInd[~isBoundary]
        innerCoords = self.cellCoords[~isBoundary]

        return boundaryCoords, innerCoords


class SCPMatrix:
    def __init__(self, cells: CellCollection):
        self.r = cells.r  # radius of each cell
        self.innerCoords = cells.innerCoords
        self.boundaryCoords = cells.boundaryCoords
        self.cellCoords = cells.cellCoords
        self.candidateRadii, self.matrix = self.getMatrix()
        self.nMOS = None

    def getMatrix(self):
        """Get the scp matrix.
        """
        nCandidateSphere = len(self.cellCoords)
        radii = np.empty(nCandidateSphere, dtype=float)
        # 计算每一个cell对应的侯选球的半径
        for i in range(nCandidateSphere):
            radii[i] = self.r + np.linalg.norm(
                self.cellCoords[i] - self.boundaryCoords, axis=1).min()
        radii = np.around(radii, decimals=6)

        # 计算SCP 0-1矩阵
        matrix = np.zeros((nCandidateSphere, nCandidateSphere), dtype=np.bool)
        for i in range(nCandidateSphere):
            dist = self.r + np.linalg.norm(
                self.cellCoords[i] - self.cellCoords, axis=1)
            dist = np.around(dist, decimals=6)
            matrix[:, i] = dist <= radii[i]  # 考虑加个松弛变量，让radii[i]包含的点更多一点？
        return radii, matrix

    def solver(self):
        """The function is to solve the scp matrix.
        """
        mosInd = set()  # multiple overlapping spheres
        matrix = self.matrix
        toBeDeletedLineMask = np.sum(matrix, axis=1) == 1
        _, columns = np.nonzero(matrix[toBeDeletedLineMask, :] == 1)
        matrix = matrix[~toBeDeletedLineMask, :]
        mosInd.update(set(columns))
        while matrix.shape[0] > 0:
            columnSum = np.sum(matrix, axis=0)
            toBeAddedColumn = np.argmax(columnSum)

            if columnSum[toBeAddedColumn] == 0:
                raise Exception(
                    "There shouldn't be a column that sums to zero in scp matrix.")

            mosInd.add(toBeAddedColumn)
            toBeDeletedLineMask = matrix[:, toBeAddedColumn] == 1
            matrix = matrix[~toBeDeletedLineMask, :]

        mosInd = list(mosInd)
        center = self.cellCoords[mosInd, :]
        radii = self.candidateRadii[mosInd]
        self.nMOS = len(mosInd)
        return center, radii

    def solverTest(self):
        center, radii = self.solver()
        for i, c in enumerate(center):
            r = radii[i]
            plotter.sphere(c, r, opacity=1.0)
        # mlab.show()


def plotTest(sand: Sand, cells: CellCollection):
    sand.visualize(voxel=True, glyph="sphere")
    # mlab.points3d(cells.cellBox, color=(0, 1, 0), opacity=0.6)
    # mlab.points3d(cells.pcaPoints[:, 0],
    #               cells.pcaPoints[:, 1],
    #               cells.pcaPoints[:, 2], color=(1, 0, 0), opacity=1.0)
    mlab.points3d(cells.cellCoords[:, 0],
                  cells.cellCoords[:, 1],
                  cells.cellCoords[:, 2], color=(0, 1, 0), opacity=1.0)
    mlab.points3d(cells.boundaryCoords[:, 0],
                  cells.boundaryCoords[:, 1],
                  cells.boundaryCoords[:, 2], color=(0, 0, 1), opacity=0.7)
    # plotter.cuboid(*cells._getCuboid(), opacity=0.5)
    mlab.xlabel("x")
    mlab.ylabel("y")
    mlab.zlabel("z")
    mlab.show()


if __name__ == "__main__":
    # datafile = r"E:\Code\VAE\data\test_set.npy"
    # data = np.load(datafile)
    # sandCube = data[46, 0]
    sandCube = np.load("../132.npy")
    sand = Sand(sandCube)
    sandPointCloud = sand.point_cloud().T
    cells = CellCollection(sandPointCloud, nX=20)
    scpMatrix = SCPMatrix(cells)
    scpMatrix.solverTest()
    plotTest(sand, cells)
