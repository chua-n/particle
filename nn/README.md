## `nn` 文件夹说明

-   `trial/`

    存放一些“试验阶段”的网络模型。比如在将网上找的开源代码真正应用到自己的数据集之前，在这里对代码进行调优、测试，以给自己训练模型用作参考。

    > `vae.py`是本人的神经网络入门之作。

-   `config/`

    存放神经模型的架构配置文件，如每一层网络的神经元数量、采用何种激活函数等，同时记录 epoch、batchsize、lr 等超参数。
