DANNet: A One-Stage Domain Adaptation Network for Unsupervised Nighttime Semantic Segmentation
========
# 主要贡献
1. **DANNNet**是第一个用于夜间语义分割的单阶段适应(one-stage adaptation)框架。这个域适应的过程是通过**GAN**实现的，设计一个共享权重的RelightNet和Semantic Segmentation Network充当GAN中的Generator，然后有两个Discriminators分别识别来自源域 **S**(数据来自ImageNet)还是目标域 **D<sub>d</sub>**(Dark Zurich-D (日间拍摄))和识别来自源域 **S**(数据来自ImageNet)还是目标域 **D<sub>n</sub>**(Dark Zurich-N (夜间拍摄))的。
2. 用daytime图像分割的结果作为对应nighttime的label来作伪监督学习，因为拍摄时间不同，只能保证静态物体(比如天空，楼房，街道等)的位置信息相同，所以此处计算Loss的时候只考虑静态对象的分类。
3. Ablation study证明了DANNet的每一个部件都是有效的。

## DANNet的三个模块
DANNet 同时进行从 **S** 到 **D<sub>d</sub>** 和从 **S** 到 **D<sub>n</sub>** 的域适应训练。它由三个不同的模块组成: 图像重光照网络、语义分割网络和两个鉴别器
![image](https://user-images.githubusercontent.com/70552149/166411973-17aa7c00-f830-4bae-9564-c3431751c7bb.png)

### RelightNet图像重光照网络
![image](https://user-images.githubusercontent.com/70552149/166411882-fcc87202-8214-4c78-a6c4-a28ba97daecf.png)
该网络的输入为三种不同的数据域：![image](https://user-images.githubusercontent.com/70552149/166438580-a3d7d34d-941f-43df-b3aa-2916127cdef3.png) 生成后的图像表示为：![image](https://user-images.githubusercontent.com/70552149/166438614-73f4504c-7030-40b3-bbad-72a2c1a4da24.png) 。所有的输入图像共享相同的权重值
1. RelightNet的作用是让来源不同的三类输入在进入分割网络前的强度分布相互靠近, 也就是 **R<sub>s</sub>**、**R<sub>td</sub>** 和 **R<sub>tn</sub>** 在强度分布上要相似, 从而使后续的语义分割网络对光照变化的敏感性降低。这个网络一共有三个Loss组成了**L<sub>light</sub>**  <br> **L<sub>light</sub>** = **α<sub>tv</sub>** **L<sub>tv</sub>** + **α<sub>exp</sub>** **L<sub>exp</sub>** + **α<sub>ssim</sub>** **L<sub>ssim</sub>**
2. 其中 **L<sub>tv</sub>** 是一种广泛应用于图像去噪和图像合成的损失函数，目的是让生成的图像噪点更小，看起来更加平滑，有利于之后的分割网络。
3. 其中 **L<sub>exp</sub>** 用来让白天和黑夜场景下的输入最终有相近的光照效果。
4. 其中 **L<sub>ssim</sub>** 是一种广泛应用在图像重建的损失函数，是为了保证生成的重光照结果R和输入I有着相同的结构信息。此处使用简化版本的SSIM。
### Semantic segmentation network 语义分割网络
论文选择了三种常用的分割网络Deeplab-v2，RefineNet和PSPNet。 采用ResNet-101 作为框架的主干。语义分割网络以**R<sub>s</sub>**、**R<sub>td</sub>** 和 **R<sub>tn</sub>** 作为网络的输入，并生成相应的预测值**P<sub>s</sub>**、**P<sub>td</sub>** 和 **P<sub>tn</sub>** 。图像重光照网络和语义分割网络共同构造了DANNet的生成器 **G**。 
1. 语义分割网络采用了三种热门的网络，分别是Deeplab-v2，RefineNet和PSPNet ，backbone全部采用ResNet-101。
2. 一次训练中只使用上述的一种网络，对于来自三个域的输入，共享网络权重。三个域输入分割的结果记作**R<sub>s</sub>**、**R<sub>td</sub>** 和 **R<sub>tn</sub>**。
3. 由于源域中不同对象类别的像素数量不平衡，占比大的类别(比如sky, road and sidewalk等)容易收敛，而占比小的类别则相反。统计数据集中每一个对象类别的像素占比，设计一个re-weighting策略。避免数值爆炸使用了**对数**，并且统计了平均值和标准差对重加权的参数进行标准化（训练时凭经验初始化了std=0.05，avg=1.0，但在消融实验中std=0.16时效果最好）。
4. 结合(3)中提到的重加权策略改进了cross-entropy loss，记作**L<sub>seg</sub>**。注意只有源域的输入有ground-truth，所以只有源域进行这个损失函数计算。
5. 来自daytime和nighttime的分割结果可以进行伪监督学习，即是用daytime图像分割的结果作为对应nighttime的label，然后进行损失函数的计算。这里只考虑静态目标类别，损失函数记作**L<sub>static</sub>**。因为两张图拍摄时可能没有完全对其，所以引入𝑝(𝑐,𝑖)来解决这个问题。公式(9)![image](https://user-images.githubusercontent.com/70552149/166455807-b29cd12f-baa0-41d1-b5d8-cbc6c57fd389.png)
中的j表示i周围的3×3坐标，c表示目标类别，**o**表示对伪标签进行ont-hot编码。
### GAN 鉴别器
鉴别器的作用是区分分割预测来说源域还是目标域。本文按照[1], 修改了[2]中的框架，利用了完全卷积的层。特别地，它包括5个卷积核大小为 4 × 4 卷积层通道数分别为 {64, 128, 256, 256, 1}。由于我们有两个目标域**T<sub>d</sub>** 和 **T<sub>n</sub>** ,我们设计了两个鉴别器来区分输出来自源域还是目标域**T<sub>d</sub>** 和识别来自源域还是目标域**T<sub>n</sub>** 的。
1. Generator **G** 由重光照网络和语义分割网络组成。
2. 有两个Discriminators，记作**D<sub>d</sub>** 和**D<sub>n</sub>** ，分别识别来自源域(数据来自CityScapes，有label)还是目标域**T<sub>d</sub>**(Dark Zurich-D (daytime))和识别来自源域(数据来自CityScapes，有label)还是目标域**T<sub>n</sub>**(Dark Zurich-N (nighttime))的。
3. 把前两个网络封装看作**G** ，**G**和**D** 交替训练。
4. 𝐺训练时的损失函数为**L<sub>total</sub>** ，为了使得分割结果**P<sub>td</sub>** 和**P<sub>tn</sub>** 与源域**P<sub>s</sub>** 的分割结果相近，增加了损失函数**L<sub>adv</sub>** ，本质是两个最小二乘损失函数的和。其中r是**P<sub>td</sub>**   和**P<sub>tn</sub>** 对应的label。
5. 𝐷训练时的损失函数为 **L<sub>d</sub>** 和 **L<sub>n</sub>** ，其中符号定义与(4.)中类似

# 参考文献
[1] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. Int. Conf. Learn. Represent.,2015. **4** 
[文章链接](https://arxiv.org/abs/1511.06434v2)
![image](https://user-images.githubusercontent.com/70552149/166624490-dc3c0ad1-2e88-4e4c-bfd4-4bf9b7eecb62.png)

[2] Yi-Hsuan Tsai, Wei-Chih Hung, Samuel Schulter, Kihyuk Sohn, Ming-Hsuan Yang, and Manmohan Chandraker. Learning to adapt structured output space for semantic segmentation. In IEEE Conf. Comput. Vis. Pattern Recog., pages 7472–7481, 2018. **2, 4, 6, 7, 8**

# 相关基础概念学习
1. [监督学习和非监督学习](https://zhuanlan.zhihu.com/p/142345604) 区别：有无预期输出
2. [迁移学习](https://www.zhihu.com/question/41979241)
