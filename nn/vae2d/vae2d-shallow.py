import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class VAE2d(nn.Module):
    def __init__(self, n_latent=32, lamb=1, lr=1e-3, batch_norm=True, mode='train'):
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
        self.conv1 = self.conv2d(1, 16, 5, 2,
                                 padding=2, bias=False)
        self.conv2 = self.conv2d(16, 32, 5, 2,
                                 padding=1, bias=False)
        self.conv3 = self.conv2d(32, 64, 5, 4,
                                 padding=1, bias=False)

        self.fc_features = 64 * 25 * 25
        self.fc1 = nn.Linear(self.fc_features, self.n_latent, bias=True)
        self.fc2 = nn.Linear(self.fc_features, self.n_latent, bias=True)
        self.fc3 = nn.Linear(self.n_latent, self.fc_features, bias=True)

        self.conv_transpose1 = self.conv_transpose2d(64, 32, 5, 4,
                                                     padding=1, output_padding=1, bias=False)
        self.conv_transpose2 = self.conv_transpose2d(32, 16, 5, 2,
                                                     padding=1, output_padding=0, bias=False)
        self.conv_transpose3 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 5, 2,
                               padding=2, output_padding=1, bias=True, padding_mode='zeros'),
            *(nn.BatchNorm2d(1, affine=self.mode),  # Pythonic! ( ͡° ͜ʖ ͡°)✧ HaHa.
              nn.Sigmoid()) if self.use_bn else (nn.Sigmoid(),)
        )

    def forward(self, x):
        """神经网络的前向计算函数，定义整个模型的执行过程：从输入到输出。

            在VAE中前向计算经过三步：
                1) 计算均值 mu 和标准差 log_sigma;
                2) 重参数技巧获取隐变量 z;
                3) 解码器生成重建颗粒 x_re."""

        mu, log_sigma = self.encoder(x)
        z = self.reparameterize(mu, log_sigma)
        x_re = self.decoder(z)
        return x_re, mu, log_sigma

    def conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
               bias=True, padding_mode='zeros'):
        if self.use_bn:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, bias=bias, padding_mode='zeros'),
                nn.BatchNorm2d(out_channels, affine=self.mode),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, bias=bias, padding_mode='zeros'),
                nn.ReLU()
            )

    def conv_transpose2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                         output_padding=0, bias=True, padding_mode='zeros'):
        if self.use_bn:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias, padding_mode='zeros'),
                nn.BatchNorm2d(out_channels, affine=self.mode),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias, padding_mode='zeros'),
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
                               self.conv3
                               )(source)
        # self.image_size_after_conv = source.shape[-2], source.shape[-1]
        source = source.view(source.shape[0], -1)
        mu = self.fc1(source)
        log_sigma = self.fc2(source)
        return mu, log_sigma

    def decoder(self, coding):
        """VAE 解码器。"""

        # names=('nSamples', 'channels', 'D', 'H', 'W')
        reconstruction = coding
        reconstruction = self.fc3(reconstruction)
        reconstruction = reconstruction.view(
            reconstruction.shape[0], -1, 25, 25)
        reconstruction = nn.Sequential(
            self.conv_transpose1,
            self.conv_transpose2,
            self.conv_transpose3
        )(reconstruction)
        return reconstruction

    def criterion(self, x_re, x, mu, log_sigma):
        """学习准则，即损失函数。"""

        loss_re = F.binary_cross_entropy(x_re, x, reduction='none')
        loss_re = torch.sum(loss_re, axis=tuple(range(1, loss_re.ndim)))
        loss_re = torch.mean(loss_re)   # scalar
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
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
        return

    def contrast(self, x):
        import matplotlib.pyplot as plt

        with torch.no_grad():
            x_re, *_ = self.forward(x)
            fig, axes = plt.subplots(1, 2)
            axes[0].set_title('Real Image')
            axes[1].set_title('Fake Image')
            # plt.imshow()似乎会自动以(0, 1)之间的浮点数绘图
            axes[0].imshow(x[0, 0].to('cpu').numpy(), cmap='gray')
            axes[1].imshow(x_re[0, 0].to('cpu').numpy(), cmap='gray')
        return fig

    def random_generate(self, coding=None, device='cpu'):
        """生成一个随机数编码，绘制 VAE 解码器解码生成的图像。

        Parameters:
        -----------
        coding(2-d torch.Tensor, optional): The given coding. If not given
            explicitly, randomly sampled from the normal distribution.
        voxel(bool, optional): Whether to draw a voxelization figure
        glyph(str, optinoal): The glyph represents a single voxel, this argument
            works only when `voxel=True`"""

        from skimage import io as skio
        if coding is None:
            coding = torch.randn(1, self.n_latent).to(
                dtype=torch.float, device=device)
        with torch.no_grad():
            fake = self.decoder(coding)
            fake *= 255
            fake = fake.to(dtype=torch.uint8, device='cpu').numpy()[0, 0]
            # fake = skio.imshow(fake, cmap='gray')
        return fake

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


def test_shape(model, source):
    vae = model
    source = source
    print('source:', source.shape)
    x = source.clone()
    x = vae.conv1(x)
    x = vae.conv2(x)
    x = vae.conv3(x)

    mu, log_sigma = vae.encoder(source)
    print('after encoder:', mu.shape, log_sigma.shape)
    z = vae.reparameterize(mu, log_sigma)
    print('size of coding:', z.shape)

    re = vae.fc3(z)
    re = re.view(re.shape[0], -1, 25, 25)
    print('before transpose:', re.shape)
    re = vae.conv_transpose1(re)
    re = vae.conv_transpose2(re)
    re = vae.conv_transpose3(re)
    print(F.binary_cross_entropy(re, source))


def main():
    import time
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io as skio

    begin = time.time()
    N_LATENT = 32
    LAMB = 1
    LR = 1e-3
    N_EPOCH = 200
    BS = 128
    # VAE2d.set_seed(314, strict=True)

    # source_path = r'D:\真实颗粒训练集\CT—400'
    # storage_path = r'D:\jupyter数据文件夹\fake\vae_Images'
    source_path = '/media/chua_n/13062557197/CT—400'
    storage_path = '/media/chua_n/资料/tmp'
    
    os.chdir(source_path)
    imgs_name = os.listdir()
    imgs_name.sort()
    sample = skio.imread(imgs_name[0])
    source = np.empty((len(imgs_name), 1, *sample.shape), dtype=sample.dtype)
    for i in range(len(imgs_name)):
        source[i, 0] = skio.imread(imgs_name[i])
    source[source == 255] = 1
    source = torch.from_numpy(source)
    dataset = TensorDataset(source)
    train_set = DataLoader(dataset, batch_size=BS, shuffle=True)
    os.chdir(storage_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    vae = VAE2d(n_latent=N_LATENT, lamb=LAMB, lr=LR).to(device)
    vae.initialize()
    optimizer = vae.optimizer()

    for epoch in range(N_EPOCH):
        for i, (x,) in enumerate(train_set):
            x = x.to(dtype=torch.float, device=device)
            x_re, mu, log_sigma = vae(x)
            loss_re, loss_kl, loss = vae.criterion(x_re, x, mu, log_sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 30 == 0 or (i + 1) == len(train_set):
                time_cost = int(time.time() - begin)
                print('Time cost so far: {}h {}min {}s'.format(
                    time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
                print("Epoch[{}/{}], Step [{}/{}], Loss_re: {:.4f}, Loss_kl: {:.4f}, Loss: {:.4f}".
                      format(epoch + 1, N_EPOCH, i + 1, len(train_set), loss_re.item(), loss_kl.item(), loss.item()))
        torch.save({
            'source_path': os.path.abspath(source_path),
            'source_size': source.shape,
            'batch_size': BS,
            'epoch': '{}/{}'.format(epoch + 1, N_EPOCH),
            'step': '{}/{}'.format(i + 1, len(train_set)),
            'n_latent': vae.n_latent,
            'lamb': vae.lamb,
            'lr': vae.lr,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_re': loss_re,
            'loss_kl': loss_kl,
            'loss': loss}, './state_dict.tar')
        
        vae.eval()
        with torch.no_grad():
            index = np.random.randint(0, len(source))
            fig = vae.contrast(
                source[index:index+1].to(dtype=torch.float, device=device))
            plt.savefig('contrast {}.png'.format(epoch+1))
            plt.close()
            for i in range(2):
                fig = vae.random_generate(device=device)
                skio.imsave(
                    'random_generate {}-{}.png'.format(epoch + 1, i + 1), fig)
        vae.train()

    time_cost = int(time.time() - begin)
    print('Total time cost: {}h {}min {}s'.format(
        time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))


if __name__ == '__main__':
    # vae = VAE2d()
    # source = torch.randn(10, 1, 402, 402)
    # test_shape(vae, source)
    main()
