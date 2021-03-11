import os
import numpy as np
from mayavi import mlab

import torch
from torch import nn
from torch import autograd
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


def compute_gradient_penalty(net_D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    device = real_samples.device
    real_samples = real_samples.detach()
    fake_samples = fake_samples.detach()
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), *[1]*(real_samples.dim()-1),
                       device=device)
    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    d_out = net_D(interpolates)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_out,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_out),
        retain_graph=True,
        create_graph=True
    )[0]
    # gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def generate(net_G: Generator, vector):
    net_G = net_G.cpu()
    net_G.eval()
    if vector.shape == (net_G.nLatent,):
        vector.unsqueeze_(dim=0)

    with torch.no_grad():
        cubes = net_G(vector)
        cubes = cubes[0, 0] if cubes.size(0) == 1 else cubes[:, 0]
    cubes = cubes.numpy()
    return cubes


def train(source_path='data/liutao/v1/particles.npz',
          xml="particle/nn/config/wgan_gp.xml",
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          img_dir="output/wgan_gp/process",
          ckpt_dir="output/wgan_gp",
          log_dir="output/wgan_gp"):

    # build train set
    hp, _ = parseConfig(xml)
    source = loadNnData(source_path, 'trainSet')
    train_set = DataLoader(TensorDataset(source),
                           batch_size=hp['bs'], shuffle=True)

    # build nn model
    net_D = Critic(xml).to(device)
    net_G = Generator(xml).to(device)

    # build optimizer
    optim_D = torch.optim.Adam(
        net_D.parameters(), lr=hp['lr'], betas=(0.5, 0.999))
    optim_G = torch.optim.Adam(
        net_G.parameters(), lr=hp['lr'], betas=(0.5, 0.999))

    # set output settings
    logger = setLogger("wgan_gp", log_dir)
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
            # 梯度惩罚
            gradient_penalty = compute_gradient_penalty(net_D, x, fake)
            # 计算loss，损失函数不取log
            w_dist = score_real - score_fake
            loss_D = -w_dist + hp['lambda']*gradient_penalty
            loss_D.backward()
            optim_D.step()

            # (2) Update G network: maximize E[D(G(z))]
            if (i + 1) % hp['iterD'] == 0 or (i + 1) == len(train_set):
                for _ in range(hp['iterG']):
                    net_G.zero_grad()
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    noise = torch.randn(
                        x.size(0), hp['nLatent'], device=device)
                    fake = net_G(noise)
                    score_G = net_D(fake).mean()
                    loss_G = -score_G
                    loss_G.backward()
                    optim_G.step()

            if (i + 1) % (5*hp['iterD']) == 0 or (i + 1) == len(train_set):
                logger.info("[Epoch {}/{}] [Step {}/{}] [loss_D: {:.4f}] [w_dist: {:.4f}] [score_G: {:.4f}]".format(
                    epoch + 1, hp["nEpoch"], i + 1, len(train_set), loss_D.item(), w_dist.item(), score_G.item()))

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
