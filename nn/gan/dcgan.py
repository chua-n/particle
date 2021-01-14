from torch import nn


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
