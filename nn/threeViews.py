import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from particle.utils import project
from particle.pipeline import Sand


class Reconstructor(nn.Module):
    """Reconstruct a particle from it's three views from x, y, z orientation.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, padding=1, bias=False),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU())
        self.conv_transpose1 = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, 4, 1, padding=0, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU())
        self.conv_transpose2 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU())
        self.conv_transpose3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU())
        self.conv_transpose4 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, 2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.conv_transpose5 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, 4, 2, padding=1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x = nn.Sequential(self.conv1,
                          self.conv2,
                          self.conv3)(x)
        x = x.view(x.size(0), 128 * 8 * 8)
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 1, 1, 1)
        x = nn.Sequential(self.conv_transpose1,
                          self.conv_transpose2,
                          self.conv_transpose3,
                          self.conv_transpose4,
                          self.conv_transpose5)(x)
        return x

    def criterion(self, y_re, y):
        """重建损失."""
        loss_re = F.binary_cross_entropy(y_re, y, reduction='none')
        loss_re = torch.sum(loss_re, axis=tuple(range(1, loss_re.ndim)))
        loss_re = torch.mean(loss_re)   # scalar
        return loss_re

    @staticmethod
    def get_projection_set(source_set):
        """source_set: torch.Tensor"""
        projection_set = torch.empty((source_set.size(0), 3, source_set.size(2), source_set.size(2)),
                                     dtype=source_set.dtype)
        for i in range(source_set.size(0)):
            for j in range(3):
                projection_set[i, j] = project(source_set[i, 0], j)
        return projection_set

    def contrast(self, x, y, voxel=False, glyph='sphere'):
        """对于输入的数据源颗粒y及其投影x，对比其原始的颗粒图像与对应的重建颗粒的图像，
        默认绘制的是三维面重建后再采用高斯模糊的图像。

        Parameters:
        -----------
        x(array_like): 3 * 64 * 64
        y(array_like): 64 * 64 * 64
        voxel(bool, optional): Whether to draw a voxel-represented figure
        glyph(str, optional): The glyph used to represent a single voxel, works only
            when `voxel=True`.
        """
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y)
            x, y = x.view(1, *x.shape), y.view(1, 1, *y.shape)
            y_re = self.forward(x)
            raw_cube = Sand(y[0, 0].detach().numpy())
            fake_cube = Sand(y_re[0, 0].detach().numpy())
            fig1 = raw_cube.visualize(figure='Original Particle',
                                      voxel=voxel, glyph=glyph)
            fig2 = fake_cube.visualize(figure='Reconstructed Particle',
                                       voxel=voxel, glyph=glyph, scale_mode='scalar')
        return fig1, fig2

    def generate(self, x, color=(0.65, 0.65, 0.65), opacity=1.0, voxel=False, glyph='sphere'):
        """
        Parameters:
        -----------
        x(array_like): 3 * 64 * 64
        voxel(bool, optional): Whether to draw a voxelization figure
        glyph(str, optinoal): The glyph represents a single voxel, this argument
            works only when `voxel=True`"""
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            x.unsqueeze_(dim=0)
            y_re = self.forward(x)
            cube = Sand(y_re[0, 0].detach().numpy())
            fig = cube.visualize(figure='Generated Particle', color=color, opacity=opacity,
                                 voxel=voxel, glyph=glyph, scale_mode='scalar')
        return fig


if __name__ == '__main__':
    import time
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    torch.manual_seed(3.14)

    LR = 0.001
    # BETA = 0.5
    N_EPOCH = 30
    BS = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_path = '../VAE/data'
    cwd = os.getcwd()
    os.chdir(source_path)
    source_train = torch.from_numpy(np.load('train_set.npy'))
    source_test = torch.from_numpy(np.load('test_set.npy'))
    os.chdir(cwd)
    projection_train = Reconstructor.get_projection_set(source_train)
    projection_test = Reconstructor.get_projection_set(source_test)
    train_set = DataLoader(
        TensorDataset(projection_train, source_train), batch_size=BS, shuffle=True)
    test_set = DataLoader(TensorDataset(
        projection_test, source_test), batch_size=2*BS, shuffle=False)

    model = Reconstructor().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    test_losses = []
    time_begin = time.time()
    for epoch in range(N_EPOCH):
        model.train()
        for i, (x, y) in enumerate(train_set):
            x = x.to(dtype=torch.float32, device=device)
            y = y.to(dtype=torch.float32, device=device)
            y_re = model(x)
            loss = model.criterion(y_re, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            if (i + 1) % 10 == 0 or (i + 1) == len(train_set):
                time_cost = int(time.time() - time_begin)
                print('Time cost so far: {}h {}min {}s'.format(
                    time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
                print("Epoch[{}/{}], Step [{}/{}], Loss_re: {:.4f}".
                      format(epoch + 1, N_EPOCH, i + 1, len(train_set), loss.item()))

        # 评估在测试集上的损失
        model.eval()
        with torch.no_grad():
            # 为啥autopep8非要把我的lambda表达式给换成def函数形式？？？
            def transfer(x): return x.to(dtype=torch.float32, device=device)
            # sum函数将其内部视为生成器表达式？？？
            test_loss = sum(model.criterion(
                model(transfer(x)), transfer(y)) for x, y in test_set)
            test_loss /= len(test_set)  # 这里取平均数
            test_losses.append(test_loss)
        time_cost = int(time.time() - time_begin)
        print('Time cost so far: {}h {}min {}s'.format(
            time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
        print('The loss in test set after {}-th epoch is: {:.4f}'.format(
            epoch + 1, test_loss))

        torch.save({  # 每轮结束保存一次模型数据
            'source_path': os.path.abspath(source_path),
            'train_set_size': source_train.shape,
            'test_set_size': source_test.shape,
            'batch_size': BS,
            'epoch': '{}/{}'.format(epoch + 1, N_EPOCH),
            'step': '{}/{}'.format(i + 1, len(train_set)),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss_re': loss.item()}, './state_dict.tar')

    time_cost = int(time.time() - time_begin)
    print('Total time cost: {}h {}min {}s'.format(
        time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))

    # Plot the training losses.
    plt.style.use('seaborn')
    plt.figure()
    plt.title("Reconstruction Loss in Train Set")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.savefig("Train Set")
    # Plot the testing losses.
    plt.figure()
    plt.title("Reconstruction Loss in Test Set")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.plot(test_losses)
    plt.savefig("Test Set")
