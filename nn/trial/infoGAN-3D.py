import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image

from particle.utils.plotter import fig2array, makeGrid

ngf = 64  # generator feature map size
ndf = 64  # discriminator feature map size


class Generator(nn.Module):
    def __init__(self, n_noise=62, n_disc=10, n_cont=2):
        super().__init__()
        self.n_latent = n_noise + n_disc + n_cont

        # n_latent * 1 * 1 * 1
        self.conv_transpose1 = nn.Sequential(
            nn.ConvTranspose3d(self.n_latent, ngf * 8, 4,
                               1, padding=0, bias=False),
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
            nn.Sigmoid())  # nn.tanh???

        # 1 * 64 * 64 * 64

    def forward(self, x):
        x = x.view(*x.shape, 1, 1, 1)
        return nn.Sequential(self.conv_transpose1,
                             self.conv_transpose2,
                             self.conv_transpose3,
                             self.conv_transpose4,
                             self.conv_transpose5)(x)


class Share(nn.Module):
    """The common part for Discriminator and net Q."""

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

    def forward(self, x):
        x = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # (ndf * 8) * 4 * 4 * 4
        self.conv5 = nn.Sequential(
            nn.Conv3d(ndf * 8, 1, 4, 1, padding=0, bias=False),
            nn.Sigmoid())
        # finally: 1 * 1 * 1 * 1

    def forward(self, x):
        x = self.conv5(x)
        return x


class Q(nn.Module):
    def __init__(self, n_disc=10, n_cont=2):
        super().__init__()
        self.n_disc = n_disc
        self.n_cont = n_cont
        # 此层暴力迁移过来的
        self.conv5 = nn.Sequential(
            nn.Conv3d(ndf * 8, 128, 4, 1, padding=0, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2))
        self.fc_disc = nn.Linear(128, self.n_disc)
        self.fc_cont_mu = nn.Linear(128, self.n_cont)
        self.fc_cont_var = nn.Linear(128, self.n_cont)

    def forward(self, x):
        """回头试试这种：

        # mu = x[:, self.n_disc:self.n_disc+self.n_cont]
        # var = x[:, self.n_disc+self.n_cont:]
        """
        x = self.conv5(x).squeeze()
        disc = self.fc_disc(x)
        cont_mu = self.fc_cont_mu(x)
        cont_var = torch.exp(self.fc_cont_var(x))
        return disc, cont_mu, cont_var


def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
            (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def sample_z(n_noise, n_disc, n_cont, batch_size, device):
    noise = torch.randn(batch_size, n_noise, device=device)
    # torch.multinomial()?
    # one-hot vectors
    disc = torch.as_tensor(np.random.multinomial(1, n_disc * [1 / n_disc], size=batch_size),
                           dtype=torch.float32, device=device)
    cont = torch.empty(batch_size, n_cont, dtype=torch.float32,
                       device=device).uniform_(-1, 1)
    z = torch.cat((noise, disc, cont), dim=1)
    return z


N_NOISE = 62
N_DISC = 6
N_CONT = 5
N_EPOCH = 10
BS = 64

LR_D = 0.0002  # 2e-4
LR_G = 0.001  # 1e-3
BETA1 = 0.5
BETA2 = 0.999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source_path = './data/train_set.npy'
source = torch.from_numpy(np.load(source_path))   # device(type='cpu')
train_set = DataLoader(TensorDataset(source), batch_size=BS, shuffle=True)

net_Share = Share().to(device)
net_D = Discriminator().to(device)
net_Q = Q(N_DISC, N_CONT).to(device)
net_G = Generator(N_NOISE, N_DISC, N_CONT).to(device)
# for net in [net_Share, net_D, net_Q, net_G]:
#     net.apply(weights_init)

optim_D = torch.optim.Adam([{"params": net_Share.parameters()}, {
                           "params": net_D.parameters()}], lr=LR_D, betas=(BETA1, BETA2))
optim_G = torch.optim.Adam([{"params": net_G.parameters()}, {
                           "params": net_Q.parameters()}], lr=LR_G, betas=(BETA1, BETA2))

criterion_D = nn.BCELoss()
criterion_disc = nn.CrossEntropyLoss()
criterion_cont = NormalNLLLoss()  # log_gaussian()

real_label = 1
fake_label = 0

# fixed latent codes
fixed_batch_size = N_DISC * 10
fixed_noise = torch.randn(fixed_batch_size, N_NOISE, device=device)
fixed_disc = torch.cat((torch.eye(N_DISC, device=device),)*10, dim=0)
cont_template = torch.repeat_interleave(torch.linspace(-1, 1, 10), N_DISC)
fixed_cont1, fixed_cont2, fixed_cont3, fixed_cont4, fixed_cont5 = [
    torch.zeros((fixed_batch_size, N_CONT), device=device) for i in range(N_CONT)]
fixed_cont_list = [fixed_cont1, fixed_cont2,
                   fixed_cont3, fixed_cont4, fixed_cont5]
for i, fixed_cont in enumerate(fixed_cont_list):
    fixed_cont[:, i] = cont_template
fixed_z1 = torch.cat((fixed_noise, fixed_disc, fixed_cont1), dim=1)
fixed_z2 = torch.cat((fixed_noise, fixed_disc, fixed_cont2), dim=1)
fixed_z3 = torch.cat((fixed_noise, fixed_disc, fixed_cont3), dim=1)
fixed_z4 = torch.cat((fixed_noise, fixed_disc, fixed_cont4), dim=1)
fixed_z5 = torch.cat((fixed_noise, fixed_disc, fixed_cont5), dim=1)
# fixed_cont1 = torch.cat(
#     (cont_template, torch.zeros_like(cont_template)), dim=1).to(device)
# fixed_cont2 = torch.cat(
#     (torch.zeros_like(cont_template), cont_template), dim=1).to(device)
# fixed_z1 = torch.cat((fixed_noise, fixed_disc, fixed_cont1), dim=1)
# fixed_z2 = torch.cat((fixed_noise, fixed_disc, fixed_cont2), dim=1)

losses_D = []
losses_G = []
losses_Info = []

time_begin = time.time()
for epoch in range(N_EPOCH):
    for i, (x,) in enumerate(train_set):
        # udpate Discriminator
        optim_D.zero_grad()
        # 判真
        x = x.to(dtype=torch.float, device=device)
        label = torch.full((x.size(0),), real_label,
                           device=device, dtype=torch.float)
        judgement_real = net_D(net_Share(x)).view(-1)
        loss_D_real = criterion_D(judgement_real, label)
        loss_D_real.backward()
        # 判假
        z = sample_z(N_NOISE, N_DISC, N_CONT, x.size(0), device)
        label.fill_(fake_label)
        fake = net_G(z)
        judgement_fake = net_D(net_Share(fake.detach())).view(-1)
        loss_D_fake = criterion_D(judgement_fake, label)
        loss_D_fake.backward()
        # 综合
        loss_D = loss_D_real + loss_D_fake
        optim_D.step()

        # update Generator and Q
        optim_G.zero_grad()
        share_out = net_Share(fake)
        # update Generator part
        judgement = net_D(share_out).view(-1)
        label.fill_(real_label)
        # treat fake data as real
        loss_G_reconstruct = criterion_D(judgement, label)
        # update Q part
        q_disc, q_cont_mu, q_cont_var = net_Q(share_out)
        disc = z[:, N_NOISE: N_NOISE + N_DISC]
        # torch.max(disc, 1)[1]是nn.CrossEntropyLoss()的target
        loss_G_disc = criterion_disc(q_disc, torch.max(disc, 1)[1])
        cont = z[:, -N_CONT:]
        # cont本采样自均匀分布，现在却又用均值和方差来衡量其与正态分布的距离？？？
        # 迷之操作？？？
        loss_G_cont = 0.1 * criterion_cont(cont, q_cont_mu, q_cont_var)
        loss_G = loss_G_reconstruct + loss_G_disc + loss_G_cont
        loss_G.backward()
        optim_G.step()

        losses_D.append(loss_D.item())
        losses_G.append(loss_G.item())
        losses_Info.append((loss_G_disc + loss_G_cont).item())

        if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
            time_cost = int(time.time() - time_begin)
            print('Time cost so far: {}h {}min {}s'.format(
                time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
            print("Epoch[{}/{}], Step [{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}, Loss_Info: {:.4f}".
                  format(epoch + 1, N_EPOCH, i + 1, len(train_set), loss_D.item(), loss_G.item(), (loss_G_disc + loss_G_cont).item()))

    with torch.no_grad():
        for i, fixed_z in enumerate([fixed_z1, fixed_z2, fixed_z3, fixed_z4, fixed_z5]):
            buffers = []
            cubes = net_G(fixed_z).to('cpu').numpy()
            for cube in cubes:
                cube = cube[0]  # discard the channel axis
                x, y, z = np.nonzero(cube)
                flatten = cube.reshape(-1)
                val = flatten[np.nonzero(flatten)]
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                plt.axis('off')
                ax.scatter(x, y, z, s=val)
                buffer = fig2array(fig)
                buffers.append(buffer)
            makeGrid(
                buffers, './output/infogan/process/{}-fixed_z{}.png'.format(epoch + 1, i + 1), nrow=N_DISC, normalize=True)

# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(losses_G, label="G")
plt.plot(losses_D, label="D")
plt.plot(losses_Info, label='Info')
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss Curve")
