import os

import torch
from mayavi import mlab
from particle.pipeline import Sand
from particle.utils.log import setLogger
from particle.utils.xml import constructOneLayer, parseConfig
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, xmlFile):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.hp = hp
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("conv3d"):
                self.add_module(
                    layerType.split('_')[-1], constructOneLayer(layerType, layerParam))

    def forward(self, x):
        judge = x
        for module in self.children():
            judge = module(judge)
        judge = judge.view(-1)
        return judge

    def weights_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                # module.weight or module.weight.data ?
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)  # no change
        return


class Generator(nn.Module):
    def __init__(self, xmlFile):
        super().__init__()
        hp, nnParams = parseConfig(xmlFile)
        self.hp = hp
        for layerType, layerParam in nnParams.items():
            if layerType.startswith("convTranspose3d"):
                self.add_module(
                    layerType.split('_')[-1], constructOneLayer(layerType, layerParam))

    def forward(self, x):
        output = x.reshape(*x.shape, 1, 1, 1)
        for module in self.children():
            output = module(output)
        return output

    def weights_init(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)  # no change
        return


def generate(net_G, vector):
    net_G = net_G.cpu()
    net_G.eval()
    if vector.shape == (net_G.hp['nLatent'],):
        vector.unsqueeze_(dim=0)

    with torch.no_grad():
        cubes = net_G(vector)
        cubes = cubes[0, 0] if cubes.size(0) == 1 else cubes[:, 0]
    return cubes


def train(net_D, net_G, train_set, device, img_dir="outout/dcgan/process", log_dir="output/log", ckpt_dir="output/dcgan/param"):
    logger = setLogger("dcgan", log_dir)
    ckpt_dir = os.path.abspath(ckpt_dir)
    img_dir = os.path.abspath(img_dir)
    assert net_D.hp == net_G.hp, "The hyperparameters in Discriminator and Generator are supposed to be the same."
    hp = net_D.hp
    logger.critical(f"\n{hp}")
    net_D = net_D.to(device)
    net_G = net_G.to(device)
    logger.critical(f"\n{net_D}")
    logger.critical(f"\n{net_G}")
    optim_D = torch.optim.Adam(
        net_D.parameters(), lr=hp['lr'], betas=(0.5, 0.999))
    optim_G = torch.optim.Adam(
        net_G.parameters(), lr=hp['lr'], betas=(0.5, 0.999))

    # 要知道：G是通过优化D来间接提升自己的，故两个网络只需一个loss criterion
    criterion = nn.BCELoss()

    # Create a batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(5, hp['nLatent'], device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    for epoch in range(hp["nEpoch"]):
        for i, (x,) in enumerate(train_set):
            x = x.to(dtype=torch.float, device=device)
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # 判真
            label = torch.full((x.size(0),), real_label,
                               device=device, dtype=torch.float)
            net_D.zero_grad()
            judgement_real = net_D(x)
            loss_D_real = criterion(judgement_real, label)
            loss_D_real.backward()
            D_x = judgement_real.mean().item()
            # 判假
            noise = torch.randn(x.size(0), hp['nLatent'], device=device)
            fake = net_G(noise)
            label.fill_(fake_label)
            judgement_fake = net_D(fake.detach())
            loss_D_fake = criterion(judgement_fake, label)
            loss_D_fake.backward()
            D_G_z1 = judgement_fake.mean().item()
            loss_D = loss_D_real + loss_D_fake
            optim_D.step()

            # (2) Update G network: maximize log(D(G(z)))
            net_G.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            judgement = net_D(fake)
            loss_G = criterion(judgement, label)
            loss_G.backward()
            D_G_z2 = judgement.mean().item()
            optim_G.step()

            if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
                logger.info("Epoch[{}/{}], Step [{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f} / {:.4f}".
                            format(epoch + 1, hp["nEpoch"], i + 1, len(train_set), loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

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
