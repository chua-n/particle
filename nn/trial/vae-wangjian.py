import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, n_latent=64, lamb=1, lr=1e-3, batch_norm=True, mode='train'):
        """VAE 模型类实例初始化函数，定义了各神经网络层的架构。

        Parameters:
        -----------
        n_latent(int): 隐变量空间的维数
        lamb(float): 损失函数中KL距离所占的权重
        lr(float): learning rate, 设定的初始的学习率
        batch_norm(bool): 是否使用批量归一化
        mode(str, 'train' or 'test'): 训练模式或测试模式（因为据说测试集时不能开启batch_norm等，会有区别）

        Note:
        -----
        按照 PyTorch 官方源码的一个示例，根据 Python 类语法，原来网络权重（和偏置）的初始化过程
            也可以放到此__init__函数中来，这样会使得在构造一个类实例的时候会自动完成你设置的初始
            化参数过程。即，现在所采用的版本中，方法initialize()的定义内容可直接放到__init__
            定义里的下方。"""

        if type(batch_norm) is not bool:
            raise TypeError(
                "The `batch_norm` parameter is expected to be a bool type.")

        if mode not in {'train', 'test'}:
            raise ValueError(
                "The `mode` parameter should be either 'train' or 'test'.")

        super().__init__()
        self.n_latent = n_latent
        self.lamb = lamb
        self.lr = lr
        self.use_bn = batch_norm
        self.mode = True if mode == 'train' else False
        self.test = not self.train  # maybe not be used
        # 1 * 200 * 200 * 200
        self.conv1 = self.conv3d(1, 16, 5, 2,
                                 padding=2, bias=False)
        # 16* 100 * 100 * 100
        self.conv2 = self.conv3d(16, 32, 5, 2,
                                 padding=2, bias=False)
        # 32 * 50 * 50 * 50
        self.conv3 = self.conv3d(32, 64, 5, 2,
                                 padding=2, bias=False)
        # 64 * 25 * 25 * 25
        self.conv4 = self.conv3d(64, 64, 5, 2,
                                 padding=2, bias=False)
        # 64 * 13 * 13 * 13
        self.conv5 = self.conv3d(64, 128, 5, 2,
                                 padding=2, bias=False)
        # 128 * 7 * 7 * 7
        self.conv6 = self.conv3d(128, 256, 5, 2,
                                 padding=2, bias=False)
        # 256 * 4 * 4 * 4
        self.conv7 = self.conv3d(256, 256, 4, 1,
                                 padding=0, bias=False)
        # 256 * 1 * 1 * 1
        self.fc1 = nn.Linear(256, self.n_latent, bias=True)
        self.fc2 = nn.Linear(256, self.n_latent, bias=True)
        self.fc3 = nn.Linear(self.n_latent, 256, bias=True)
        self.conv_transpose1 = self.conv_transpose3d(256, 256, 4, 1,
                                                     padding=0, output_padding=0, bias=False)
        self.conv_transpose2 = self.conv_transpose3d(256, 128, 5, 2,
                                                     padding=2, output_padding=0, bias=False)
        self.conv_transpose3 = self.conv_transpose3d(128, 64, 5, 2,
                                                     padding=2, output_padding=0, bias=False)
        self.conv_transpose4 = self.conv_transpose3d(64, 64, 5, 2,
                                                     padding=2, output_padding=0, bias=False)
        self.conv_transpose5 = self.conv_transpose3d(64, 32, 5, 2,
                                                     padding=2, output_padding=1, bias=False)
        self.conv_transpose6 = self.conv_transpose3d(32, 16, 5, 2,
                                                     padding=2, output_padding=1, bias=False)
        self.conv_transpose7 = nn.Sequential(
            nn.ConvTranspose3d(16, 1, 5, 2,
                               padding=2, output_padding=1, bias=True, padding_mode='zeros'),
            *(nn.BatchNorm3d(1, affine=self.mode),  # Pythonic! ( ͡° ͜ʖ ͡°)✧ HaHa.
              nn.Sigmoid()) if self.use_bn else (nn.Sigmoid(),)
        )

    def forward(self, x):
        """神经网络的前向计算函数，定义整个模型的执行过程：从输入到输出。

            在VAE中前向计算经过三步：
                1) 计算均值 mu 和标准差 log_sigma;
                2) 重参数技巧获取隐变量 z;
                3) 解码器生成重建颗粒 x_re."""

        print(x.size())
        print(x.device)
        mu, log_sigma = self.encoder(x)
        z = self.reparameterize(mu, log_sigma)
        x_re = self.decoder(z)
        return x_re, mu, log_sigma

    def conv3d(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
               bias=True, dilation=1, groups=1, padding_mode='zeros'):
        """Be sensitive to the various parameters in convolution layers!

        Parameters:
        -----------
        dilation(int or tuple): In PyTorch, `dilation` actually means "dilation rate" 
            depictedin various literatures, which means there are dilation-1 spaces 
            inserted between kernel elements such that `dilation=1` corresponds to a 
            regular convolution. That's why the default value of `dilation` is 1.
        padding_mode(str, 'zeros'): 只能设为'zeros', 猜测这是为了后续丰富功能选项设置的向后
            兼容的API接口。
        groups: ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        """

        if self.use_bn:
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, bias=bias,
                          dilation=dilation, groups=groups, padding_mode=padding_mode),
                nn.BatchNorm3d(out_channels, affine=self.mode),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, bias=bias,
                          dilation=dilation, groups=groups, padding_mode=padding_mode),
                nn.ReLU()
            )

    def conv_transpose3d(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                         output_padding=0, bias=True, groups=1, dilation=1, padding_mode='zeros'):
        """Be sensitive to the various parameters in transposed convolution layers!

        Parameters:
        -----------
        padding(int or tuple): In PyTorch, the `padding` argument effectively adds 
            `dilation * (kernel_size - 1) - padding` amount of zero padding to both 
            sizes of the input.
        output_padding(int or tuple): Additional size added to ONE side of each dimension
            in the output shape. Default: 0
        dilation(int or tuple): In PyTorch, `dilation` actually means "dilation rate" 
            depictedin various literatures, which means there are dilation-1 spaces 
            inserted between kernel elements such that `dilation=1` corresponds to a 
            regular convolution. That's why the default value of `dilation` is 1.
        padding_mode(str, 'zeros'): 只能设为'zeros', 猜测这是为了后续丰富功能选项设置的向后
            兼容的API接口。
        groups: ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        """
        if self.use_bn:
            return nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias,
                                   groups=groups, dilation=dilation, padding_mode='zeros'),
                nn.BatchNorm3d(out_channels, affine=self.mode),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias,
                                   groups=groups, dilation=dilation, padding_mode='zeros'),
                nn.ReLU()
            )

    def reparameterize(self, mu, log_sigma):
        """重参数技巧，进行再参数化。

        Note:
        -----
        log_sigma 原来用 long_val/2 表示，不过在这里和后续 loss 中的数学公式两者等价，
            实际也没产生什么影响"""

        sigma = torch.exp(log_sigma)
        epsilon = torch.randn_like(sigma)
        coding = mu + sigma * epsilon   # 即隐变量z
        return coding

    def encoder(self, source):
        """VAE 编码器。

            注意，这里没有编码成最终coding，而是构建了均值和标准差。"""
        source = nn.Sequential(self.conv1,
                               self.conv2,
                               self.conv3,
                               self.conv4,
                               self.conv5,
                               self.conv6,
                               self.conv7,
                               )(source)
        source = source.view(source.shape[0], -1)
        mu = self.fc1(source)
        log_sigma = self.fc2(source)
        return mu, log_sigma

    def decoder(self, coding):
        """VAE 解码器。"""

        # names=('nSamples', 'channels', 'D', 'H', 'W')
        reconstruction = coding
        reconstruction = self.fc3(reconstruction)
        reconstruction = reconstruction.view(*reconstruction.shape, 1, 1, 1)
        reconstruction = nn.Sequential(
            self.conv_transpose1,
            self.conv_transpose2,
            self.conv_transpose3,
            self.conv_transpose4,
            self.conv_transpose5,
            self.conv_transpose6,
            self.conv_transpose7,
        )(reconstruction)
        return reconstruction

    def criterion(self, x_re, x, mu, log_sigma):
        """学习准则，即损失函数。"""

        loss_re = F.binary_cross_entropy(x_re, x, reduction='none')
        loss_re = torch.sum(loss_re, axis=tuple(range(1, loss_re.ndim)))
        loss_re = torch.mean(loss_re)   # scalar
        # loss_kl如下公式通常写作负数形式，但经测试写成如下形式没问题
        loss_kl = 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(log_sigma) - log_sigma - 1,
                                  axis=tuple(range(1, mu.ndim)))
        loss_kl = torch.mean(loss_kl)   # scalar
        loss = loss_re + self.lamb * loss_kl    # scalar
        return loss_re, loss_kl, loss

    def optimizer(self):
        """优化器，即采用的优化算法。"""

        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
            x_re, *_ = self.forward(x)
            raw_cube = Sand(x[0, 0].detach().numpy())
            fake_cube = Sand(x_re[0, 0].detach().numpy())
            fig1 = raw_cube.visualize(figure='Original Particle',
                                      voxel=voxel, glyph=glyph)
            fig2 = fake_cube.visualize(figure='Reconstructed Particle',
                                       voxel=voxel, glyph=glyph, scale_mode='scalar')
        return fig1, fig2

    def random_generate(self, coding=None, color=(0.65, 0.65, 0.65), opacity=1.0, voxel=False, glyph='sphere'):
        """开启eval()模式，生成一个随机数编码，绘制 VAE 解码器解码生成的图像。若要继续训练模型，
            记得在此函数调用后将模型设为train()模式。

        Parameters:
        -----------
        coding(2-d torch.Tensor, optional): The given coding. If not given 
            explicitly, randomly sampled from the normal distribution.
        voxel(bool, optional): Whether to draw a voxelization figure
        glyph(str, optinoal): The glyph represents a single voxel, this argument
            works only when `voxel=True`"""

        from particle.pipeline import Sand
        if coding is None:
            coding = torch.randn(1, self.n_latent)
        self.eval()
        with torch.no_grad():
            fake = self.decoder(coding)
            fake = fake.detach().numpy()[0, 0]
            fake = Sand(fake)
            fig = fake.visualize(figure='Randomly Generated Particle', color=color, opacity=opacity,
                                 voxel=voxel, glyph=glyph, scale_mode='scalar')
        return fig

    @staticmethod
    def set_seed(seed, strict=False):
        """Set the random number seed, so that the results may be reproduced
            if you want.

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


if __name__ == "__main__":

    #device = torch.device("cuda")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # 64: 41GB
    # 32: 21GB
    x = torch.randn(20, 1, 200, 200, 200, dtype=torch.float32)

    vae = VAE().cuda()
    lossFn = vae.criterion
    vae = nn.DataParallel(vae, device_ids=[0, 1])
    x = x.cuda()
    # torch.save("xxxxx.pt", vae.state_dict())
    print(vae)
    # x = x.to(device)
    # model = vae.to(device)

    optim = torch.optim.Adam(vae.parameters(), 1e-3)

    xRe, mu, logSigma = vae(x)
    # print(xRe.shape)
    loss_re, loss_kl, loss = lossFn(xRe, x, mu, logSigma)
    optim.zero_grad()
    loss.backward()
    optim.step()
