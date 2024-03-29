# 项目说明

以显微 CT 扫描的土颗粒图像为研究对象，主要涉及 CT 图像处理（从原始图像中提取出三维颗粒的过程）、三维颗粒几何形态分析、深度学习模型生成颗粒、离散元 clump 生成算法。

## 文件（夹）说明

-   `clump/`: 离散元 clump 生成算法
-   `nn/`: 神经网络模型
-   `utils/`: 一些工具类/工具函数
-   ~~`mindboggle/`~~: 网上的开源代码，与计算 Zernike 矩相关，在 2021/4/2 重构项目时已被删除

## 进度记录

-   wgan: wgan_cp 与 wgan_gp 总是训练不出来，无法收敛。
-   class `Sand`: 位姿归一化方法未校正。
-   关于当前训练出的`TVSNet`模型，发现其从三视图重建颗粒时，其重构出来的颗粒可能并非所想要的那个形状，它对三张视图的排列序列并不完全鲁棒。目前试验表明，对于脑海中想像出的颗粒及其正、侧、俯视图，在将其拼装成 `3*64*64` 的数组时必须按照俯、正、侧视图的顺序，这要才能重构出与目标形状相同的颗粒。
