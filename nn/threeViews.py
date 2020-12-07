import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from particle.utils import project, parseConfig, constructOneLayer
from particle.pipeline import Sand


class Reconstructor(nn.Module):
    """Reconstruct a particle from it's three views from x, y, z orientation.
    """

    def __init__(self, xmlFile):
        super().__init__()
        nnParams = parseConfig(xmlFile)
        self.convBlock = nn.Sequential()
        self.fcBlock = nn.Sequential()
        self.convTransposeBlock = nn.Sequential()
        self.fcBlockInFeatures = None
        self.fcBlockOutFeatures = None
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("fc"):
                if self.fcBlockInFeatures is None:
                    self.fcBlockInFeatures = nnParams[layerType]["in_features"]
                self.fcBlockOutFeatures = nnParams[layerType]["out_features"]
                self.fcBlock.add_module(
                    layerType.split("_")[-1], constructOneLayer(layerType, layerParam))
            elif layerType.startswith("convTranspose"):
                self.convTransposeBlock.add_module(
                    layerType.split("_")[-1], constructOneLayer(layerType, layerParam))
            else:
                self.convBlock.add_module(
                    layerType.split("_")[-1], constructOneLayer(layerType, layerParam))

    def forward(self, x):
        x = self.convBlock(x)
        x = x.view(x.size(0), self.fcBlockInFeatures)
        x = self.fcBlock(x)
        x = x.view(x.size(0), self.fcBlockOutFeatures, 1, 1, 1)
        x = self.convTransposeBlock(x)
        return x

    def criterion(self, y_re, y):
        """重建损失."""
        loss_re = F.binary_cross_entropy(y_re, y, reduction='none')
        loss_re = torch.sum(loss_re, axis=tuple(range(1, loss_re.ndim)))
        loss_re = torch.mean(loss_re)   # scalar
        return loss_re

    @staticmethod
    def get_projection_set(source_set):
        """source_set: torch.Tensor"""
        projection_set = torch.empty((source_set.size(0), 3, source_set.size(2), source_set.size(2)),
                                     dtype=source_set.dtype)
        for i in range(source_set.size(0)):
            for j in range(3):
                projection_set[i, j] = project(source_set[i, 0], j)
        return projection_set

    def contrast(self, x, y, voxel=False, glyph='sphere'):
        """对于输入的数据源颗粒y及其投影x，对比其原始的颗粒图像与对应的重建颗粒的图像，
        默认绘制的是三维面重建后再采用高斯模糊的图像。

        Parameters:
        -----------
        x(array_like): 3 * 64 * 64
        y(array_like): 64 * 64 * 64
        voxel(bool, optional): Whether to draw a voxel-represented figure
        glyph(str, optional): The glyph used to represent a single voxel, works only
            when `voxel=True`.
        """
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y)
            x, y = x.view(1, *x.shape), y.view(1, 1, *y.shape)
            y_re = self.forward(x)
            raw_cube = Sand(y[0, 0].detach().numpy())
            fake_cube = Sand(y_re[0, 0].detach().numpy())
            fig1 = raw_cube.visualize(figure='Original Particle',
                                      voxel=voxel, glyph=glyph)
            fig2 = fake_cube.visualize(figure='Reconstructed Particle',
                                       voxel=voxel, glyph=glyph, scale_mode='scalar')
        return fig1, fig2

    def generate(self, x, color=(0.65, 0.65, 0.65), opacity=1.0, voxel=False, glyph='sphere'):
        """
        Parameters:
        -----------
        x(array_like): 3 * 64 * 64
        voxel(bool, optional): Whether to draw a voxelization figure
        glyph(str, optinoal): The glyph represents a single voxel, this argument
            works only when `voxel=True`"""
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            x.unsqueeze_(dim=0)
            y_re = self.forward(x)
            cube = Sand(y_re[0, 0].detach().numpy())
            fig = cube.visualize(figure='Generated Particle', color=color, opacity=opacity,
                                 voxel=voxel, glyph=glyph, scale_mode='scalar')
        return fig
