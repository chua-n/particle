import os

import torch
from mayavi import mlab
from particle.pipeline import Sand
from particle.utils.log import setLogger
from particle.utils.config import constructOneLayer, parseConfig
from torch import nn


class Critic(nn.Module):
    def __init__(self, xml):
        super().__init__()
        hp, nnParams = parseConfig(xml)
        self.hp = hp
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
        self.hp = hp
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("convT-"):
                self.add_module(
                    layerType.split('-')[-1], constructOneLayer(layerType, layerParam))

    def forward(self, vector):
        output = vector.reshape(*vector.shape, 1, 1, 1)
        for module in self.children():
            output = module(output)
        return output


def generate(net_G, vector):
    net_G = net_G.cpu()
    net_G.eval()
    if vector.shape == (net_G.hp['nLatent'],):
        vector.unsqueeze_(dim=0)

    with torch.no_grad():
        cubes = net_G(vector)
        cubes = cubes[0, 0] if cubes.size(0) == 1 else cubes[:, 0]
    return cubes


def train(net_D, net_G, train_set, device, img_dir="output/wgan_cp/process", ckpt_dir="output/wgan_cp", log_dir="output/wgan_cp"):
    logger = setLogger("wgan", log_dir)
    ckpt_dir = os.path.abspath(ckpt_dir)
    img_dir = os.path.abspath(img_dir)
    assert net_D.hp == net_G.hp, "The hyperparameters in Discriminator and Generator are supposed to be the same."
    hp = net_D.hp
    logger.critical(f"\n{hp}")
    net_D = net_D.to(device)
    net_G = net_G.to(device)
    logger.critical(f"\n{net_D}")
    logger.critical(f"\n{net_G}")

    optim_D = torch.optim.RMSprop(net_D.parameters(), lr=hp['lr'])
    optim_G = torch.optim.RMSprop(net_G.parameters(), lr=hp['lr'])

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

            # (2) Update G network: maximize log(D(G(z)))
            if (i + 1) % hp['iterD'] == 0 or (i + 1) == len(train_set):
                for _ in hp['iterG']:
                    net_G.zero_grad()
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    noise = torch.randn(x.size(0), hp['nLatent'], device=device)
                    fake = net_G(noise)
                    score_G = net_D(fake).mean(0).view(1)
                    loss_G = -score_G
                    loss_G.backward()
                    optim_G.step()

            if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
                logger.info("[Epoch {}/{}] [Step {}/{}] [w_dist: {:.4f}] [score_G: {:.4f}]".format(
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
