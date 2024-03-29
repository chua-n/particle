import os
from mayavi import mlab

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from particle.pipeline import Sand
from particle.utils.dirty import loadNnData
from particle.utils.log import setLogger
from particle.utils.config import constructOneLayer, parseConfig


class Critic(nn.Module):
    def __init__(self, xml):
        super().__init__()
        _, nnParams = parseConfig(xml)
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("conv-"):
                self.add_module(
                    layerType.split('-')[-1], constructOneLayer(layerType, layerParam))

    def forward(self, x):
        score = x
        for module in self.children():
            score = module(score)
        score = score.view(score.size(0))
        return score


class Generator(nn.Module):
    def __init__(self, xml):
        super().__init__()
        hp, nnParams = parseConfig(xml)
        self.nLatent = hp['nLatent']
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("convT-"):
                self.add_module(
                    layerType.split('-')[-1], constructOneLayer(layerType, layerParam))

    def forward(self, vector):
        output = vector.reshape(*vector.shape, 1, 1, 1)
        for module in self.children():
            output = module(output)
        return output


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
          xml="particle/nn/config/wgan_cp.xml",
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          img_dir="output/wgan_cp/process",
          ckpt_dir="output/wgan_cp",
          log_dir="output/wgan_cp"):

    # build train set
    hp, _ = parseConfig(xml)
    source = loadNnData(source_path, 'trainSet')
    train_set = DataLoader(TensorDataset(source),
                           batch_size=hp['bs'], shuffle=True)

    # build nn model
    net_D = Critic(xml).to(device)
    net_G = Generator(xml).to(device)

    # build optimizer
    optim_D = torch.optim.RMSprop(net_D.parameters(), lr=hp['lr'])
    optim_G = torch.optim.RMSprop(net_G.parameters(), lr=hp['lr'])

    # set output settings
    logger = setLogger("wgan_cp", log_dir)
    ckpt_dir = os.path.abspath(ckpt_dir)
    img_dir = os.path.abspath(img_dir)
    logger.critical(f"\n{hp}")
    logger.critical(f"\n{net_D}")
    logger.critical(f"\n{net_G}")

    # Create a batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(5, hp['nLatent'], device=device)

    for epoch in range(hp["nEpoch"]):
        for i, (x,) in enumerate(train_set):
            x = x.to(dtype=torch.float, device=device)
            # (1) Update D network: maximize E[D(x)] - E[D(G(z))], where E represents the math expectation
            # 判真
            net_D.zero_grad()
            score_real = net_D(x).mean()
            # 判假
            noise = torch.randn(x.size(0), hp['nLatent'], device=device)
            fake = net_G(noise)
            score_fake = net_D(fake.detach()).mean()
            # 计算loss，损失函数不取log
            w_dist = score_real - score_fake
            loss_D = -w_dist
            loss_D.backward()
            optim_D.step()

            with torch.no_grad():
                for p in net_D.parameters():
                    p.clamp_(-hp['clip'], hp['clip'])

            # (2) Update G network: maximize E[D(G(z))]
            if (i + 1) % hp['iterD'] == 0 or (i + 1) == len(train_set):
                for _ in range(hp['iterG']):
                    net_G.zero_grad()
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    noise = torch.randn(
                        x.size(0), hp['nLatent'], device=device)
                    fake = net_G(noise)
                    score_G = net_D(fake).mean(0).view(1)
                    loss_G = -score_G
                    loss_G.backward()
                    optim_G.step()

            if (i + 1) % (5*hp['iterD']) == 0 or (i + 1) == len(train_set):
                logger.info("[Epoch {}/{}] [Step {}/{}] [w_dist {:.4f}] [score_G {:.4f}]".format(
                    epoch + 1, hp["nEpoch"], i + 1, len(train_set), w_dist.item(), score_G.item()))

        # 每轮结束保存一次模型参数
        torch.save({
            'discriminator_state_dict': net_D.state_dict(),
            'generator_state_dict': net_G.state_dict(),
            'optim_D_state_dict': optim_D.state_dict(),
            'optim_G_state_dict': optim_G.state_dict()}, os.path.join(ckpt_dir, 'state_dict.tar'))
        logger.info(f"Model checkpoint has been stored in {ckpt_dir}.")

        net_G.eval()
        with torch.no_grad():
            cubes = net_G(fixed_noise).to('cpu').numpy()
            for i, cube in enumerate(cubes):
                cube = cube[0]
                sand = Sand(cube)
                sand.visualize(voxel=True, glyph='point', scale_mode='scalar')
                mlab.outline()
                mlab.axes()
                mlab.savefig(os.path.join(img_dir, f'{epoch + 1}-{i + 1}.png'))
                mlab.close()
        net_G.train()
    logger.info("Train finished!")
