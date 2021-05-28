import os

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from particle.utils.dirty import loadNnData
from particle.utils.log import setLogger
from particle.utils.config import parseConfig, constructOneLayer


def setSeed(seed, strict=False):
    """Set the random number seed for PyTorch, so that the results may be 
    reproduced if you want.

    Parameters:
    -----------
    seed(int): The desired seed.
    strict(bool, optional): If False, you can reproduce your work on CPU
        completely, but on GPU it still has subtle difference. If True, 
        even on GPU your result will repeat, but note that this may reduce
        GPU computational efficiency."""

    if type(strict) is not bool:
        raise TypeError("`strict` must be bool type.")
    torch.manual_seed(seed)
    if strict:
        print("Warning: Set `strict=True` may reduce computational efficiency of your GPUs. "
              "Be cautious to do this!")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return


class Encoder(nn.Module):
    """VAE 编码器。注意VAE的编码器不是直接进行“编码”的，而是构建了编码的均值和标准差。
    """

    def __init__(self, xmlFile):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.nLatent = hp['nLatent']
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("Encoder-"):
                self.add_module('-'.join(layerType.split('-')[1:]),
                                constructOneLayer(layerType, layerParam))
                self._out_channels = layerParam['out_channels']
        self.fc1 = nn.Linear(self._out_channels, self.nLatent, bias=True)
        self.fc2 = nn.Linear(self._out_channels, self.nLatent, bias=True)

    def forward(self, x):
        for name, module in self.named_children():
            if name.startswith("conv-"):
                x = module(x)
        # x = x.view(x.size(0), -1)
        x = x.view(*x.shape[:2])
        mu = self.fc1(x)
        logSigma = self.fc2(x)
        return mu, logSigma


class Decoder(nn.Module):
    """VAE解码器。"""

    def __init__(self, xmlFile):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.nLatent = hp['nLatent']
        self._out_features = nnParams['Decoder-convT-1']['in_channels']
        self.fc = nn.Linear(self.nLatent, self._out_features)
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("Decoder-"):
                self.add_module('-'.join(layerType.split('-')[1:]),
                                constructOneLayer(layerType, layerParam))

    def forward(self, coding):
        reconstruction = self.fc(coding)
        reconstruction = reconstruction.view(*reconstruction.shape, 1, 1, 1)
        for name, module in self.named_children():
            if name.startswith("convT-"):
                reconstruction = module(reconstruction)
        return reconstruction


class Vae(nn.Module):
    def __init__(self, xmlFile):
        super().__init__()
        hp, _ = parseConfig(xmlFile)
        self.lamb = hp['lambda']
        self.encoder = Encoder(xmlFile)
        self.decoder = Decoder(xmlFile)

    def forward(self, x):
        """神经网络的前向计算函数，定义整个模型的执行过程：从输入到输出。

            在VAE中前向计算经过三步：
                1) 计算均值 mu 和标准差 logSigma;
                2) 再参数化获取隐变量 z;
                3) 解码器生成重建颗粒 xRe."""

        mu, logSigma = self.encoder(x)
        z = self.reparameterize(mu, logSigma)
        xRe = self.decoder(z)
        return xRe, mu, logSigma

    def reparameterize(self, mu, logSigma):
        """再参数化技巧。

        Note:
        -----
        logSigma 原来用 logVal/2 表示，不过在这里和后续 loss 中的数学公式两者等价，实际不产生影响。
        """
        sigma = torch.exp(logSigma)
        epsilon = torch.randn_like(sigma)
        coding = mu + sigma * epsilon  # 即隐变量z
        return coding

    def criterion(self, xRe, x, mu, logSigma):
        """学习准则，即损失函数。"""

        loss_re = F.binary_cross_entropy(xRe, x, reduction='none')
        loss_re = torch.sum(loss_re, axis=tuple(range(1, loss_re.ndim)))
        loss_re = torch.mean(loss_re)   # scalar
        # loss_kl如下公式通常写作负数形式，但经测试写成如下形式没问题
        loss_kl = 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(logSigma) - logSigma - 1,
                                  axis=tuple(range(1, mu.ndim)))
        loss_kl = torch.mean(loss_kl)   # scalar
        loss = loss_re + self.lamb * loss_kl    # scalar
        return loss_re, loss_kl, loss

    def initialize(self):
        """参数初始化，即用特定的数值来初始化神经网络的各权重和偏置。"""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
        return

    def contrast(self, x, voxel=False, glyph='sphere'):
        """开启eval()模式，对于输入的数据源颗粒x，对比其原始的颗粒图像与对应的重建颗粒的图像，默认
            绘制的是三维面重建后再采用高斯模糊的图像。若要继续训练模型，记得函数调用后将模型恢复为
            train()模式。

        Parameters:
        -----------
        voxel(bool, optional): Whether to draw a voxel-represented figure
        glyph(str, optional): The glyph used to represent a single voxel, works only
            when `voxel=True`.
        """

        from particle.pipeline import Sand
        self.eval()
        with torch.no_grad():
            xRe, *_ = self.forward(x)
            rawCube = Sand(x[0, 0].detach().numpy())
            fakeCube = Sand(xRe[0, 0].detach().numpy())
            fig1 = rawCube.visualize(figure='Original Particle',
                                     voxel=voxel, glyph=glyph)
            fig2 = fakeCube.visualize(figure='Reconstructed Particle',
                                      voxel=voxel, glyph=glyph, scale_mode='scalar')
        return fig1, fig2

    def generate(self, coding=None, thrd: float = 0.5):
        """根据给定的编码向量生成颗粒。若不给定编码，随机生成一个颗粒。若要继续训练模型，
        记得在此函数调用后将模型设为train()模式。

        Parameters:
        -----------
        coding(1-d torch.Tensor-vector or a batch of torch.Tensor-vectors, optional):
            The given coding. If not given explicitly, randomly sampled from the normal distribution.

        return:
        -------
        cubes(np.array): (64, 64, 64) or (*, 64, 64, 64)
        """

        decoder = self.decoder.cpu()
        if coding is None:
            coding = torch.randn(1, decoder.nLatent)
        if coding.shape == (decoder.nLatent,):
            coding.unsqueeze_(dim=0)
        self.eval()
        with torch.no_grad():
            cubes = decoder(coding)
            cubes = cubes[0, 0] if cubes.size(0) == 1 else cubes[:, 0]
        if thrd is not None:
            cubes[cubes <= thrd] = 0
            cubes[cubes > thrd] = 1
            cubes = cubes.type(torch.uint8)
        cubes = cubes.numpy()
        return cubes


def train(sourcePath='data/liutao/v1/particles.npz',
          xml="particle/nn/config/vae.xml",
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          log_dir="output/vae",
          ckpt_dir="output/vae"):
    # build train set & test set
    hp, _ = parseConfig(xml)
    trainSet = loadNnData(sourcePath, 'trainSet')
    trainSet = DataLoader(TensorDataset(trainSet),
                          batch_size=hp['bs'], shuffle=True)
    testSet = loadNnData(sourcePath, 'testSet')
    testSet = DataLoader(TensorDataset(testSet),
                         batch_size=hp['bs']*2, shuffle=False)

    # build and initilize VAE model
    vae = Vae(xml).to(device)
    vae.initialize()
    optim = torch.optim.Adam(vae.parameters(), lr=hp['lr'])
    lossFn = vae.criterion

    # set output settings
    logger = setLogger("vae", log_dir)
    ckpt_dir = os.path.abspath(ckpt_dir)
    logger.critical(f"\n{hp}")
    logger.critical(f"\n{vae}")

    lastTestLoss = float("inf")
    for epoch in range(hp['nEpoch']):
        vae.train()
        for i, (x,) in enumerate(trainSet):
            x = x.to(dtype=torch.float, device=device)
            xRe, mu, logSigma = vae(x)
            loss_re, loss_kl, loss = lossFn(xRe, x, mu, logSigma)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if (i + 1) % 10 == 0 or (i + 1) == len(trainSet):
                logger.info("[Epoch {}/{}] [Step {}/{}] [loss_re {:.4f}] [loss_kl {:.4f}] [loss {:.4f}]".
                            format(epoch + 1, hp["nEpoch"], i + 1, len(trainSet), loss_re.item(), loss_kl.item(), loss.item()))
        vae.eval()
        logger.info("Model is running on the test set...")
        with torch.no_grad():
            testLoss_re = testLoss_kl = testLoss = 0
            for (x,) in testSet:
                x = x.to(dtype=torch.float, device=device)
                xRe, mu, logSigma = vae(x)
                lossRes = lossFn(xRe, x, mu, logSigma)
                testLoss_re += lossRes[0]
                testLoss_kl += lossRes[1]
                testLoss += lossRes[2]
            testLoss_re /= len(testSet)
            testLoss_kl /= len(testSet)
            testLoss /= len(testSet)
            logger.info("loss in test set: [testLoss_re {:.4f}] [testLoss_kl {:.4f}] [testLoss {:.4f}]".format(
                testLoss_re, testLoss_kl, testLoss))
        # 确认是否保存模型参数
        if testLoss < lastTestLoss:
            torch.save(vae.state_dict(), os.path.join(
                ckpt_dir, 'state_dict.pt'))
            logger.info(f"Model checkpoint has been stored in {ckpt_dir}.")
            lastTestLoss = testLoss
        else:
            torch.save(vae.state_dict(), os.path.join(
                ckpt_dir, 'state_dict-overfit.pt'))
            logger.warning("The model may be overfitting!")
    logger.info("Train finished!")


if __name__ == "__main__":
    # vae = Vae("./particle/nn/config/vae.xml")
    # print(vae)
    # x = torch.rand(100, 1, 64, 64, 64)
    # assert vae(x)[0].size() == x.size()
    torch.manual_seed(3.14)
    train()
