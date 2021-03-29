# clump 包的说明

此包里的各个模块是离散元 clump 建模的各种算法。

1. distTrans.py 是自主提出的基于距离变换的 clump 建模算法；
2. bubblePack.py 是 PFC 的 buble packing 算法，但由于未能实现“受限制的 Delaunay 四面体剖分”而复现失败；
3. mgh.py 是清华大学徐文杰团队提出的 clump 建模算法，但应该有地方实现不准确，因为这里复现的版本与其论文展示的效果有差异，暂时懒得调试了。
4. diagramIn2d/ 是为方便展示 clump 生成过程的 2d 算法版，只为在论文中画图使用。
