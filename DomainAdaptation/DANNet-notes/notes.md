DANNet: A One-Stage Domain Adaptation Network for Unsupervised Nighttime Semantic Segmentation
========
# 主要贡献
1. **DANNNet**是第一个用于夜间语义分割的单阶段适应(one-stage adaptation)框架。这个域适应的过程是通过**GAN**实现的，设计一个共享权重的RelightNet和Semantic Segmentation Network充当GAN中的Generator，然后有两个Discriminators分别识别来自源域 **S**(数据来自ImageNet)还是目标域 **D<sub>d</sub>**(Dark Zurich-D (日间拍摄))和识别来自源域 **S**(数据来自ImageNet)还是目标域 **D<sub>n</sub>**(Dark Zurich-N (夜间拍摄))的。
2. 用daytime图像分割的结果作为对应nighttime的label来作伪监督学习，因为拍摄时间不同，只能保证静态物体(比如天空，楼房，街道等)的位置信息相同，所以此处计算Loss的时候只考虑静态对象的分类。
3. Ablation study证明了DANNet的每一个部件都是有效的。

## DANNet的三个模块
### RelightNet
1. RelightNet的作用是让来源不同的三类输入在进入分割网络前的强度分布相互靠近，也就是 **R<sub>s</sub>**、**R<sub>td</sub>** 和 **R<sub>tn</sub>** 在强度分布上要相似。这个网络一共有三个Loss组成了**L<sub>light</sub>**  <br> **L<sub>light</sub>** = **α<sub>tv</sub>** **L<sub>tv</sub>** + **α<sub>exp</sub>** **L<sub>exp</sub>** + **α<sub>ssim</sub>** **L<sub>ssim</sub>**
2. 其中 **L<sub>tv</sub>** 是一种广泛应用于图像去噪和图像合成的损失函数，目的是让生成的图像噪点更小，看起来更加平滑，有利于之后的分割网络。
3. 其中 **L<sub>exp</sub>** 用来让白天和黑夜场景下的输入最终有相近的光照效果。
4. 其中 **L<sub>ssim</sub>** 是一种广泛应用在图像重建的损失函数，是为了保证生成的重光照结果R和输入I有着相同的结构信息。此处使用简化版本的SSIM。
# 相关基础概念学习
1. [监督学习和非监督学习](https://zhuanlan.zhihu.com/p/142345604) 区别：有无预期输出
2. 
