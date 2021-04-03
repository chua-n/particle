import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from particle.utils.log import setLogger
from particle.utils.config import parseConfig, constructOneLayer
from particle.utils.dirty import loadNnData, project
from particle.pipeline import Sand


def getProjections(sourceSet: torch.Tensor):
    projections = torch.empty((sourceSet.size(0), 3, sourceSet.size(2), sourceSet.size(2)),
                              dtype=sourceSet.dtype)
    for i in range(sourceSet.size(0)):
        for j in range(3):
            projections[i, j] = project(sourceSet[i, 0], j)
    return projections


class TVSDataset(Dataset):
    def __init__(self, dataset, transform=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        cubeImage = self.dataset[idx]
        if self.transform is not None:
            cubeImage = self.transform(cubeImage)
        cube = cubeImage[0]
        projection = torch.empty(3, cube.size(-1), cube.size(-1),
                                 dtype=cube.dtype, device=cube.device)
        for i in range(3):
            projection[i] = project(cube, i)
        return projection, cubeImage


class TVSNet(nn.Module):
    """Reconstruct a particle from it's three views from x, y, z orientation.
    """

    def __init__(self, xmlFile):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.hp = hp
        self.convBlock = nn.Sequential()
        self.fcBlock = nn.Sequential()
        self.convTransposeBlock = nn.Sequential()
        self.fcBlockInFeatures = None
        self.fcBlockOutFeatures = None
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("fc-"):
                if self.fcBlockInFeatures is None:
                    self.fcBlockInFeatures = nnParams[layerType]["in_features"]
                self.fcBlockOutFeatures = nnParams[layerType]["out_features"]
                self.fcBlock.add_module(
                    layerType.split("-")[-1], constructOneLayer(layerType, layerParam))
            elif layerType.startswith("convT-"):
                self.convTransposeBlock.add_module(
                    layerType.split("-")[-1], constructOneLayer(layerType, layerParam))
            else:
                self.convBlock.add_module(
                    layerType.split("-")[-1], constructOneLayer(layerType, layerParam))

    def forward(self, x):
        x = self.convBlock(x)
        x = x.view(x.size(0), self.fcBlockInFeatures)
        x = self.fcBlock(x)
        x = x.view(x.size(0), self.fcBlockOutFeatures, 1, 1, 1)
        x = self.convTransposeBlock(x)
        return x

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


class VaeEncoder(nn.Module):
    def __init__(self, xmlFile):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.nLatent = hp['nLatent']
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("Encoder-"):
                self.add_module('-'.join(layerType.split('-')[1:]),
                                constructOneLayer(layerType, layerParam))
                self._out_channels = layerParam['out_channels']
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(p=0.2)  # 考虑加dropout到底合不合适
        self.fc = nn.Linear(self._out_channels, 2*self.nLatent, bias=True)

    def forward(self, x):
        for name, module in self.named_children():
            if name.startswith("conv-"):
                x = module(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(*x.shape[:2])
        x = self.fc(x)
        midInd = x.size(1)//2
        mu, logSigma = x[:, :midInd], x[:, midInd:]
        return mu, logSigma


class VaeDecoder(nn.Module):
    def __init__(self, xmlFile):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.nLatent = hp['nLatent']
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("Decoder-"):
                self.add_module('-'.join(layerType.split('-')[1:]),
                                constructOneLayer(layerType, layerParam))

    def forward(self, coding):
        coding = coding.view(*coding.shape, 1, 1, 1)
        reconstruction = coding
        for name, module in self.named_children():
            if name.startswith("convT-"):
                reconstruction = module(reconstruction)
        return reconstruction


class VaeTVSNet(nn.Module):
    """Three Views Stereo Net. Reconstruct a particle from it's 
    three views from x, y, z orientation.
    """
    _bceLossFn = nn.BCELoss(reduction="sum")

    def __init__(self, xmlFile):
        super().__init__()
        hp, _ = parseConfig(xmlFile)
        self.lamb = hp['lambda']
        self.encoder = VaeEncoder(xmlFile)
        self.decoder = VaeDecoder(xmlFile)

    def forward(self, x):
        mu, logSigma = self.encoder(x)
        z = self.reparameterize(mu, logSigma)
        xRe = self.decoder(z)
        return xRe, mu, logSigma

    def reparameterize(self, mu, logSigma):
        sigma = torch.exp(logSigma)
        epsilon = torch.randn_like(sigma)
        coding = mu + sigma * epsilon
        return coding

    def lossFn(self, x, xRe, mu, logSigma):
        lossRe = self._bceLossFn(xRe, x).div_(x.size(0))
        lossKL = 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(logSigma) - logSigma - 1,
                                 axis=tuple(range(1, mu.ndim)))
        lossKL = torch.mean(lossKL)
        loss = lossRe + self.lamb * lossKL
        return lossRe, lossKL, loss

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
        return

    def contrast(self, x, y, **kwargs):
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
        status = self.training
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y)
            x, y = x.view(1, *x.shape), y.view(1, 1, *y.shape)
            yRe, *_ = self.forward(x)
            rawCube = Sand(y[0, 0].detach().numpy())
            fakeCube = Sand(yRe[0, 0].detach().numpy())
            fakeCube.cube[fakeCube.cube <= 0.5] = 0
            fakeCube.cube[fakeCube.cube > 0.5] = 1
            fig1 = rawCube.visualize(figure='Original Particle', **kwargs)
            fig2 = fakeCube.visualize(
                figure='Reconstructed Particle', **kwargs)
        self.training = status
        return fig1, fig2

    def generate(self, x, **kwargs):
        """
        Parameters:
        -----------
        x(array_like): 3 * 64 * 64"""
        status = self.training
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            x.unsqueeze_(dim=0)
            yRe, *_ = self.forward(x)
            cube = Sand(yRe[0, 0].numpy())
            fig = cube.visualize(figure='Generated Particle', **kwargs)
        self.training = status
        return fig


def train(sourcePath="data/liutao/v1/particles.npz",
          xml="particle/nn/config/tvsnet.xml",
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          logDir="output/tvsnet",
          ckptDir="output/tvsnet",
          kernel=None):
    # kernel参数指定TVSNet模型的内核采用vae模式还是自己编写的模式
    assert kernel in ("vae", None)

    # build train set & test set
    hp, _ = parseConfig(xml)
    trainSet = loadNnData(sourcePath, 'trainSet')
    trainSet = DataLoader(TVSDataset(trainSet, transform=transforms.RandomRotation(hp["rotation"])),
                          batch_size=hp['bs'], shuffle=True, drop_last=True)
    testSet = loadNnData(sourcePath, "testSet")
    testSet = DataLoader(TVSDataset(testSet),
                         batch_size=hp['bs']*2, shuffle=False)

    # build and initilize TVSNet model
    if kernel == "vae":
        model = VaeTVSNet(xml).to(device)
        model.initialize()
        lossFn = model.lossFn
    else:
        model = TVSNet(xml).to(device)
        bceLossFn = nn.BCELoss(reduction="sum")
        def lossFn(ipt, tgt): return bceLossFn(ipt, tgt)/ipt.size(0)

    # set optimizer
    optim = torch.optim.Adam(model.parameters(), hp['lr'])

    # set output settings
    logger = setLogger("TVSNet", logDir)
    ckptDir = os.path.abspath(ckptDir)
    logger.critical(f"\n{hp}")
    logger.critical(f"\n{model}")

    lastTestLoss = float("inf")
    for epoch in range(hp["nEpoch"]):
        model.train()
        for i, (x, y) in enumerate(trainSet):
            x = x.to(dtype=torch.float32, device=device)
            y = y.to(dtype=torch.float32, device=device)
            if kernel == "vae":
                lossRe, lossKL, loss = lossFn(y, *model(x))
                optim.zero_grad()
                loss.backward()
                optim.step()
                if (i + 1) % 10 == 0 or (i + 1) == len(trainSet):
                    logger.info("[Epoch {}/{}] [Step {}/{}] [lossRe {:.4f}] [lossKL {:.4f}] [loss {:.4f}]".
                                format(epoch+1, hp["nEpoch"], i+1, len(trainSet), lossRe.item(), lossKL.item(), loss.item()))
            else:
                yRe = model(x)
                loss = lossFn(yRe, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if (i + 1) % 10 == 0 or (i + 1) == len(trainSet):
                    logger.info("[Epoch {}/{}] [Step {}/{}] [lossRe {:.4f}]".
                                format(epoch+1, model.hp["nEpoch"], i+1, len(trainSet), loss.item()))

        # 评估在测试集上的损失
        model.eval()
        logger.info("Model is running on the test set...")
        with torch.no_grad():
            # 为啥autopep8非要把我的lambda表达式给换成def函数形式......
            def transfer(x): return x.to(dtype=torch.float32, device=device)
            # sum函数将其内部视为生成器表达式？？？
            if kernel == "vae":
                testLoss = sum(lossFn(transfer(y), *model(transfer(x)))[-1]
                               for (x, y) in testSet)
            else:
                testLoss = sum(lossFn(model(transfer(x)), transfer(y))
                               for (x, y) in testSet)
            testLoss /= len(testSet)  # 这里取平均数
        logger.info(f"loss in test set: [testLoss {testLoss:.4f}]")
        # 确认是否保存模型参数
        if testLoss < lastTestLoss:
            torch.save(model.state_dict(), os.path.join(
                ckptDir, 'state_dict.pt'))
            logger.info(f"Model checkpoint has been stored in {ckptDir}.")
            lastTestLoss = testLoss
        else:
            torch.save(model.state_dict(), os.path.join(
                ckptDir, 'state_dict-overfit.pt'))
            logger.warning("The model may be overfitting!")
    logger.info("Train finished!")


if __name__ == "__main__":
    torch.manual_seed(3.14)
    train(kernel=None)
    # train(kernel="vae", xml="particle/nn/config/tvsnet-vae-deep.xml")
