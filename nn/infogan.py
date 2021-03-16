import os
import numpy as np
from mayavi import mlab

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from particle.pipeline import Sand
from particle.utils.dirty import loadNnData
from particle.utils.config import parseConfig, constructOneLayer
from particle.utils.log import setLogger
from particle.utils.plotter import makeGrid


class Generator(nn.Module):
    def __init__(self, xmlFile):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.nLatent = hp['nNoise'] + hp['nDisc'] + hp['nCont']
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("Generator-"):
                self.add_module(
                    layerType.split('-')[-1], constructOneLayer(layerType, layerParam))

    def forward(self, x):
        x = x.view(*x.shape, 1, 1, 1)
        output = x
        for module in self.children():
            output = module(output)
        return output


class Share(nn.Module):
    """The common part for Discriminator and net Q."""

    def __init__(self, xmlFile):
        super().__init__()
        _, nnParams = parseConfig(xmlFile)
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("Share-"):
                self.add_module(
                    layerType.split('-')[-1], constructOneLayer(layerType, layerParam))

    def forward(self, x):
        output = x
        for module in self.children():
            output = module(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, xmlFile):
        super().__init__()
        _, nnParams = parseConfig(xmlFile)
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("Discriminator-"):
                self.add_module(
                    layerType.split('-')[-1], constructOneLayer(layerType, layerParam))

    def forward(self, x):
        output = x
        for module in self.children():
            output = module(output)
        output = output.view(output.size(0))
        return output


class Q(nn.Module):
    def __init__(self, xmlFile):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.nDisc = hp['nDisc']
        self.nCont = hp['nCont']

        # 此层暴力迁移过来的
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("Q-"):
                # self.add_module(
                #     layerType.split('-')[-1], constructOneLayer(layerType, layerParam))
                self.conv = constructOneLayer(layerType, layerParam)
                self._out_channels = layerParam['out_channels']

        self.disc = nn.Linear(self._out_channels, self.nDisc)
        self.cont_mu = nn.Linear(self._out_channels, self.nCont)
        self.cont_var = nn.Linear(self._out_channels, self.nCont)

    def forward(self, x):
        """一个想法：

        回头试试这种：
        mu = x[:, self.nDisc:self.nDisc+self.nCont]
        var = x[:, self.nDisc+self.nCont:]
        """
        x = self.conv(x).squeeze()
        disc = self.disc(x)
        cont_mu = self.cont_mu(x)
        cont_var = torch.exp(self.cont_var(x))
        return disc, cont_mu, cont_var


def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


class NormalNLLLoss:
    """Calculate the negative log likelihood of normal distribution. 

    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
            (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def sample_z(bs, nNoise, nDisc, nCont, device):
    noise = torch.randn(bs, nNoise, device=device)
    # torch.multinomial()?
    # one-hot vectors
    disc = torch.as_tensor(np.random.multinomial(1, nDisc * [1 / nDisc], size=bs),
                           dtype=torch.float32, device=device)
    cont = torch.empty(bs, nCont, dtype=torch.float32,
                       device=device).uniform_(-1, 1)
    z = torch.cat((noise, disc, cont), dim=1)
    return z


def get_fixed_z(nNoise, nDisc, nCont, device):
    """Get fixed latent vector codes to display the influence of variables in different dimensions
    on the shape of generated particle.

    NOTE:
    -------
        First of all, all batches of z will share the same noise codes. Secondly, within a single batch,
    each z-vector shares the same continuous codes, but differs on the discrete codes. Therefore, when 
    we have `nCont` number of continuous dims in a z-vector, we will return `nCont` batches.
    """
    bs = nDisc * 10  # 乘10是为了展示连续性变量的影响
    fixed_noise = torch.randn(bs, nNoise, device=device)
    fixed_disc = torch.cat((torch.eye(nDisc, device=device),)*10, dim=0)
    cont_template = torch.repeat_interleave(torch.linspace(-1, 1, 10), nDisc)
    fixed_cont_list = [torch.zeros((bs, nCont), device=device)
                       for i in range(nCont)]
    # fixed_cont1, fixed_cont2, fixed_cont3, fixed_cont4, fixed_cont5 = [
    #     torch.zeros((bs, nCont), device=device) for i in range(nCont)]
    # fixed_cont_list = [fixed_cont1, fixed_cont2,
    #                    fixed_cont3, fixed_cont4, fixed_cont5]
    # fixed_z1 = torch.cat((fixed_noise, fixed_disc, fixed_cont1), dim=1)
    # fixed_z2 = torch.cat((fixed_noise, fixed_disc, fixed_cont2), dim=1)
    # fixed_z3 = torch.cat((fixed_noise, fixed_disc, fixed_cont3), dim=1)
    # fixed_z4 = torch.cat((fixed_noise, fixed_disc, fixed_cont4), dim=1)
    # fixed_z5 = torch.cat((fixed_noise, fixed_disc, fixed_cont5), dim=1)
    for i, fixed_cont in enumerate(fixed_cont_list):
        fixed_cont[:, i] = cont_template
    fixed_z_list = [torch.cat((fixed_noise, fixed_disc, fixed_cont), dim=1)
                    for fixed_cont in fixed_cont_list]
    return fixed_z_list
    # fixed_cont1 = torch.cat(
    #     (cont_template, torch.zeros_like(cont_template)), dim=1).to(device)
    # fixed_cont2 = torch.cat(
    #     (torch.zeros_like(cont_template), cont_template), dim=1).to(device)
    # fixed_z1 = torch.cat((fixed_noise, fixed_disc, fixed_cont1), dim=1)
    # fixed_z2 = torch.cat((fixed_noise, fixed_disc, fixed_cont2), dim=1)


def generate(net_G: Generator, vector: torch.Tensor, thrd: float = 0.5):
    net_G = net_G.cpu()
    net_G.eval()
    if vector.shape == (net_G.nLatent,):
        vector.unsqueeze_(dim=0)

    with torch.no_grad():
        cubes = net_G(vector)
        cubes = cubes[0, 0] if cubes.size(0) == 1 else cubes[:, 0]
    if thrd is not None:
        cubes[cubes <= thrd] = 0
        cubes[cubes > thrd] = 1
        cubes = cubes.type(torch.uint8)
    cubes = cubes.numpy()
    return cubes


def train(source_path='data/liutao/v1/particles.npz',
          xml="particle/nn/config/infogan.xml",
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          img_dir="output/infogan/process",
          log_dir="output/infogan",
          ckpt_dir="output/infogan"):

    # build train set
    hp, _ = parseConfig(xml)
    source = loadNnData(source_path, 'trainSet')
    train_set = DataLoader(TensorDataset(
        source), batch_size=hp['bs'], shuffle=True)

    # build nn model
    net_Share = Share(xml).to(device)
    net_D = Discriminator(xml).to(device)
    net_Q = Q(xml).to(device)
    net_G = Generator(xml).to(device)
    # for net in [net_Share, net_D, net_Q, net_G]:
    #     net.apply(weights_init)

    # build optimizer and loss function
    optim_D = torch.optim.Adam([{"params": net_Share.parameters()}, {"params": net_D.parameters()}],
                               lr=hp['lrD'], betas=(0.5, 0.999))
    optim_G = torch.optim.Adam([{"params": net_G.parameters()}, {"params": net_Q.parameters()}],
                               lr=hp['lrG'], betas=(0.5, 0.999))
    criterion_D = nn.BCELoss()
    criterion_disc = nn.CrossEntropyLoss()
    criterion_cont = NormalNLLLoss()  # log_gaussian()

    # set output settings
    logger = setLogger("infogan", log_dir)
    ckpt_dir = os.path.abspath(ckpt_dir)
    img_dir = os.path.abspath(img_dir)
    logger.critical(f"\n{hp}")
    logger.critical(f"\n{net_Share}")
    logger.critical(f"\n{net_D}")
    logger.critical(f"\n{net_Q}")
    logger.critical(f"\n{net_G}")

    # fixed latent codes
    fixed_z_list = get_fixed_z(hp['nNoise'], hp['nDisc'], hp['nCont'], device)

    real_label = 1
    fake_label = 0
    for epoch in range(hp['nEpoch']):
        for i, (x,) in enumerate(train_set):
            # udpate Discriminator
            optim_D.zero_grad()
            # 判真
            x = x.to(dtype=torch.float, device=device)
            label = torch.full((x.size(0),), real_label,
                               device=device, dtype=torch.float)
            judgement_real = net_D(net_Share(x))
            loss_D_real = criterion_D(judgement_real, label)
            loss_D_real.backward()
            # 判假
            z = sample_z(x.size(0), hp['nNoise'],
                         hp['nDisc'], hp['nCont'], device)
            label.fill_(fake_label)
            fake = net_G(z)
            judgement_fake = net_D(net_Share(fake.detach()))
            loss_D_fake = criterion_D(judgement_fake, label)
            loss_D_fake.backward()
            # 综合
            loss_D = loss_D_real + loss_D_fake
            optim_D.step()

            # update Generator and Q
            if (i + 1) % hp['iterD'] == 0 or (i + 1) == len(train_set):
                for _ in range(hp['iterG']):
                    optim_G.zero_grad()
                    share_out = net_Share(fake)
                    # update Generator part
                    judgement = net_D(share_out)
                    label.fill_(real_label)
                    # treat fake data as real
                    loss_reconstruct = criterion_D(judgement, label)
                    # update Q part
                    q_disc, q_cont_mu, q_cont_var = net_Q(share_out)
                    disc = z[:, hp['nNoise']:hp['nNoise']+hp['nDisc']]
                    # torch.max(disc, 1)[1]是nn.CrossEntropyLoss()的target
                    loss_disc = criterion_disc(q_disc, torch.max(disc, 1)[1])
                    cont = z[:, -hp['nCont']:]
                    # cont本采样自均匀分布，现在却又用均值和方差来衡量其与正态分布的距离？？？
                    # 迷之操作？？？
                    loss_cont = 0.1*criterion_cont(cont, q_cont_mu, q_cont_var)
                    loss_info = loss_disc + loss_cont
                    loss_G = loss_reconstruct + loss_info
                    loss_G.backward()
                    optim_G.step()

            if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
                logger.info("[Epoch {}/{}] [Step {}/{}] [loss_D {:.4f}] [loss_G {:.4f}] [loss_Info {:.4f}]".
                            format(epoch+1, hp['nEpoch'], i+1, len(train_set), loss_D.item(), loss_G.item(), loss_info.item()))

        # 每轮结束保存一次模型参数
        torch.save({
            'netD_state_dict': net_D.state_dict(),
            'netG_state_dict': net_G.state_dict(),
            'netShare_state_dict': net_Share.state_dict(),
            'netQ_state_dict': net_Q.state_dict(),
            'optim_D_state_dict': optim_D.state_dict(),
            'optim_G_state_dict': optim_G.state_dict()}, os.path.join(ckpt_dir, 'state_dict.tar'))
        logger.info(f"Model checkpoint has been stored in {ckpt_dir}.")

        if (epoch+1) % hp['plotFrequency'] == 0 or (epoch+1) == hp['nEpoch']:
            net_G.eval()
            with torch.no_grad():
                for i, fixed_z in enumerate(fixed_z_list):
                    batchImage = []
                    cubes = net_G(fixed_z).to('cpu').numpy()
                    for cube in cubes:
                        cube = cube[0]  # discard the channel axis
                        sand = Sand(cube)
                        sand.visualize(voxel=True, glyph='point')
                        # sand.visualize(realistic=False)
                        mlab.outline()
                        img = mlab.screenshot()
                        mlab.close()
                        batchImage.append(img)
                    makeGrid(batchImage, os.path.join(img_dir, f'{epoch+1}-fixed_z{i+1}.png'),
                             nrow=len(batchImage)//hp['nDisc'], normalize=True)
            net_G.train()
    logger.info("Train finished!")


if __name__ == "__main__":
    torch.manual_seed(3.14)
    train()
