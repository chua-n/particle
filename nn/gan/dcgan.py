import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


n_latent = 100  # latent vector
ngf = 64  # generator feature map size
ndf = 64  # discriminator feature map size


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 * 64 * 64 * 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, ndf, 4, 2, padding=1, bias=False),
            # nn.BatchNorm3d(ndf),  # why have no BN layer
            nn.LeakyReLU(0.2))  # nn.LeakyReLU(0.2, inplace=True)

        # ndf * 32 * 32 * 32
        self.conv2 = nn.Sequential(
            nn.Conv3d(ndf, ndf * 2, 4, 2, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2))

        # (ndf * 2) * 16 * 16 * 16
        self.conv3 = nn.Sequential(
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2))

        # (ndf * 4) * 8 * 8 * 8
        self.conv4 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2))

        # (ndf * 8) * 4 * 4 * 4
        self.conv5 = nn.Sequential(
            nn.Conv3d(ndf * 8, 1, 4, 1, padding=0, bias=False),
            nn.Sigmoid())

        # finally: 1 * 1 * 1 * 1

    def forward(self, x):
        return nn.Sequential(self.conv1,
                             self.conv2,
                             self.conv3,
                             self.conv4,
                             self.conv5)(x)

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
    def __init__(self):
        super().__init__()

        # n_latent * 1 * 1 * 1
        self.conv_transpose1 = nn.Sequential(
            nn.ConvTranspose3d(n_latent, ngf * 8, 4, 1, padding=0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU())

        # (ngf * 8) * 4 * 4 * 4
        self.conv_transpose2 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, padding=1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU())

        # (ngf * 4) * 8 * 8 * 8
        self.conv_transpose3 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, padding=1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU())

        # (ngf * 2) * 16 * 16 * 16
        self.conv_transpose4 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, padding=1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU())

        # ngf * 32 * 32 * 32
        self.conv_transpose5 = nn.Sequential(
            nn.ConvTranspose3d(ngf, 1, 4, 2, padding=1, bias=False),
            nn.Tanh())

        # 1 * 64 * 64 * 64

    def forward(self, x):
        return nn.Sequential(self.conv_transpose1,
                             self.conv_transpose2,
                             self.conv_transpose3,
                             self.conv_transpose4,
                             self.conv_transpose5)(x)

    def weights_init(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)  # no change
        return


if __name__ == "__main__":
    import time
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    torch.manual_seed(3.14)

    LR = 0.0002
    BETA = 0.5
    N_EPOCH = 10
    BS = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_path = '../VAE/data/all.npy'
    source = torch.from_numpy(np.load(source_path))   # device(type='cpu')
    train_set = DataLoader(TensorDataset(source), batch_size=BS, shuffle=True)

    net_D = Discriminator().to(device)
    net_G = Generator().to(device)

    net_D.weights_init()
    net_G.weights_init()

    optim_D = torch.optim.Adam(net_D.parameters(), lr=LR, betas=(BETA, 0.999))
    optim_G = torch.optim.Adam(net_G.parameters(), lr=LR, betas=(BETA, 0.999))

    # 要知道：G是通过优化D来间接提升自己的，故两个网络只需一个loss criterion
    criterion = nn.BCELoss()

    # Create a batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(5, n_latent, 1, 1, 1, device=device)
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    begin = time.time()
    for epoch in range(N_EPOCH):
        for i, (x,) in enumerate(train_set):
            x = x.to(dtype=torch.float, device=device)
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # 判真
            label = torch.full((x.size(0),), real_label, device=device)
            net_D.zero_grad()
            judgement_real = net_D(x).view(-1)
            loss_D_real = criterion(judgement_real, label)
            loss_D_real.backward()
            D_x = judgement_real.mean().item()
            # 判假
            noise = torch.randn(x.size(0), n_latent, 1, 1, 1, device=device)
            fake = net_G(noise)
            label.fill_(fake_label)
            judgement_fake = net_D(fake.detach()).view(-1)
            loss_D_fake = criterion(judgement_fake, label)
            loss_D_fake.backward()
            D_G_z1 = judgement_fake.mean().item()
            loss_D = loss_D_real + loss_D_fake
            optim_D.step()

            # (2) Update G network: maximize log(D(G(z)))
            net_G.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            judgement = net_D(fake).view(-1)
            loss_G = criterion(judgement, label)
            loss_G.backward()
            D_G_z2 = judgement.mean().item()
            optim_G.step()

            if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
                time_cost = int(time.time() - begin)
                print('Time cost so far: {}h {}min {}s'.format(
                    time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
                print("Epoch[{}/{}], Step [{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f} / {:.4f}".
                      format(epoch + 1, N_EPOCH, i + 1, len(train_set), loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

        # 每轮结束保存一次模型参数
        torch.save({
            'source_size': source.shape,
            'batch_size': BS,
            'epoch': '{}/{}'.format(epoch + 1, N_EPOCH),
            'step': '{}/{}'.format(i + 1, len(train_set)),
            'n_latent': n_latent,
            'discriminator_state_dict': net_D.state_dict(),
            'generator_state_dict': net_G.state_dict(),
            'optim_D_state_dict': optim_D.state_dict(),
            'optim_G_state_dict': optim_G.state_dict(),
            'loss_D': loss_D,
            'loss_G': loss_G,
            'D(x)': D_x,
            'D(G(z))': "{:.4f} / {:.4f}".format(D_G_z1, D_G_z2)}, './state_dict.tar')

        net_G.eval()
        with torch.no_grad():
            cubes = net_G(fixed_noise).to('cpu').numpy()
            for i, cube in enumerate(cubes):
                cube = cube[0]
                x, y, z = np.nonzero(cube)
                flatten = cube.reshape(-1)
                val = flatten[np.nonzero(flatten)]
                ax = plt.axes(projection='3d')
                ax.scatter(x, y, z, s=val)
                plt.axis('off')
                plt.savefig('{}-{}.png'.format(epoch + 1, i + 1), dpi=200)
        net_G.train()

    # 以下为查看当前生成效果
    # checkpoint = torch.load('state_dict_20200206.tar')
    # net_G.load_state_dict(checkpoint['generator_state_dict'])
    # net_G.eval()
    # with torch.no_grad():
    #     vec = torch.randn(1, n_latent, 1, 1, 1, device=device)
    #     cube = net_G(vec).to('cpu').numpy()[0, 0]
    #     x, y, z = np.nonzero(cube)
    #     flatten = cube.reshape(-1)
    #     val = flatten[np.nonzero(flatten)]
    #     # mlab.points3d(x, y, z, val)
    #     # mlab.show()
    #     ax = plt.axes(projection='3d')
    #     # plot voxels
    #     # cube[cube > 0.7] = 1
    #     # cube[cube <= 0.7] = 0
    #     # ax.voxels(cube)
    #     ax.scatter(x, y, z, s=val)
    #     plt.axis('off')
    #     plt.show()
