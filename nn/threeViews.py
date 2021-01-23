import os
import torch
from torch import nn
import torch.nn.functional as F

from particle.utils.log import setLogger
from particle.utils.config import parseConfig, constructOneLayer
from particle.utils.dirty import project
from particle.pipeline import Sand


class Reconstructor(nn.Module):
    """Reconstruct a particle from it's three views from x, y, z orientation.
    """

    def __init__(self, xmlFile, log_dir="output/log/", ckpt_dir='output/threeViews'):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.hp = hp
        self.logger = setLogger("threeViews", log_dir)
        self.ckpt_dir = ckpt_dir
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


def get_projection_set(source_set):
    """source_set: torch.Tensor"""
    projection_set = torch.empty((source_set.size(0), 3, source_set.size(2), source_set.size(2)),
                                 dtype=source_set.dtype)
    for i in range(source_set.size(0)):
        for j in range(3):
            projection_set[i, j] = project(source_set[i, 0], j)
    return projection_set


def train(model: Reconstructor, train_set, test_set, device):
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=model.hp['lr'])

    logger = model.logger
    logger.critical(f"\n{model}")
    ckpt_dir = os.path.abspath(model.ckpt_dir)

    losses = []
    test_losses = []
    for epoch in range(model.hp["nEpoch"]):
        model.train()
        for i, (x, y) in enumerate(train_set):
            x = x.to(dtype=torch.float32, device=device)
            y = y.to(dtype=torch.float32, device=device)
            y_re = model(x)
            loss = model.criterion(y_re, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
                logger.info("Epoch[{}/{}], Step [{}/{}], Loss_re: {:.4f}".
                            format(epoch+1, model.hp["nEpoch"], i+1, len(train_set), loss.item()))

        # 评估在测试集上的损失
        model.eval()
        with torch.no_grad():
            # 为啥autopep8非要把我的lambda表达式给换成def函数形式？？？
            def transfer(x): return x.to(dtype=torch.float32, device=device)
            # sum函数将其内部视为生成器表达式？？？
            test_loss = sum(model.criterion(
                model(transfer(x)), transfer(y)) for x, y in test_set)
            test_loss /= len(test_set)  # 这里取平均数
            test_losses.append(test_loss)
        logger.info("The loss in test set after {}-th epoch is: {:.4f}".format(
            epoch + 1, test_loss))
        ckpt_name = f"state_dict_{test_loss}.pt" if test_loss < 7300 else "state_dict.pt"
        torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))
        logger.info(f"Model checkpoint has been stored in {ckpt_dir}.")

    logger.info("Train finished!")
    return losses, test_losses
