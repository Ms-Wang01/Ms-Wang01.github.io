# Deep Reinforcement Learning Assisted Federated Learning Algorithm for Data Management of IIoT

## IEEE Transactions on Industrial Informatics 中科院一区
## 深度强化学习辅助的联邦学习
## 背景
1. 工业物联网（IIoT）规模扩大，需要对大量用户数据进行管理。并且隐私数据通常容易受到异构网络、异构设备或恶意软件的攻击。
2. 联邦学习通过将中心服务器对原始数据的直接访问与模型训练分离，有效地保持数据的异构性，减少模型训练的偏差。
3. 强化学习(DRL)辅助FL框架是降低模型聚合的通信成本的一种方式。

## 主要贡献
1. 针对物联网设备因产生大量异构和私有数据而难以管理的问题，采用FL方法对这些数据进行训练和管理。
2. 本文提出了一个DRL辅助的语言学习框架。基于深度确定性策略梯度 (DDPG)的 DRL算法主要用于选择数据质量高的 IIoT 设备节点，以提高模型聚合速率，降低通信成本。
3. 我们分析了工业物联网设备数据的特征，然后使用 MNIST, Fashion MNIST, and CIFAR10 (IID and non-IID) 数据集进行实验评估。验证了基于DRL的FL技术处理IIoT数据的有效性。

## FL在工业物联网中的应用问题描述
IIoT设备在使用本地数据进行模型训练后，需要将本地模型上传到中央服务器，中央服务器会对全局模型进行优化。
1. IIoT设备使用本地数据进行本地计算，同时最小化预先定义的经验风险函数，然后将计算的权重更新到无线 网络接入点。
2. 无线网络接入点收集IIoT设备更新的权重，并访问FL单元以生成全局模型。
3. FL将模型训练的输出重新分配给设备，IIoT设备使用全局模型进行新一轮的局部训练。
4. 重复上述步骤 2 和 3，直到损失函数收敛或达到最大迭代次数。

可以用于FL

<img width="457" alt="image" src="https://user-images.githubusercontent.com/70552149/178522555-aded887b-0b3b-4a01-a59e-e8cb72e3270c.png">


## DRL辅助下FL算法的实现
<img width="921" alt="image" src="https://user-images.githubusercontent.com/70552149/178523434-f40be3a6-7d24-4720-9946-6bf141f7923f.png">
1. Initialization stage: 中心服务器先对工业物联网设备的连接请求进行评估。之后，中心服务器从连接的IIoT设备中随机抽取用户设备子集参加本轮培训。训练结束后，向每个选定的IIoT设备发送一个全局模型θt。
2. Training stage: 所选工业物联网设备采用本地设备数据全局训练模型，每次迭代后得到全局模型。
3. Aggregation stage:所有选定的IIoT设备将本地训练模型上传到中央服务器，然后中央服务器更新并训练新的全局模型。之后，中央服务器向新选定的工业物联网设备集合发布新的全局模型。重复上述过程，直到损失函数收敛或达到最大迭代次数。

### 选择特定的工业物联网设备的操作（调度策略）
1. 随机调度策略:在每一轮通信中，中心服务器随机选取N个相关的IIoT设备进行参数更新。同时，在中心服务器和选定的用户设备之间建立专用子通道，传输训练参数。
2. 循环调度策略:中心服务器将所有IIoT设备划分为G组。每次选择一个组建立一个信道进行通信，每次通信过程中循环更新模型参数。
3. 比例公平策略:在每一轮通信中，按照如下计算方法从C个相关的IIoT设备中选择N个:
<img width="285" alt="image" src="https://user-images.githubusercontent.com/70552149/178549318-2f19ebcc-47b2-46c7-a822-8d72f32203cf.png">

###  基于 DRL 的物联网设备节点选择
节点选择方法参考了这篇文章：
Y. Lu, X. Huang, K. Zhang, S. Maharjan, and Y. Zhang, “Blockchain em- powered asynchronous federated learning for secure data sharing in inter- net of vehicles,” IEEE Trans. Veh. Technol., vol. 69, no. 4, pp. 4298–4311, Apr. 2020.
1. 首先给出了工业物联网设备节点选择的成本指标。FL算法中节点选择的总代价由训练时间代价和训练质量代价两部分组成。
2. 工业物联网设备的节点选择是一个组合优化问题。我们将其建模为MDP，用M = {S, a, PA, CA}表示。其中，S表示IIoT设备节点的状态空间，A表示动作空间，PA表示采取的动作的状态变化概率，CA表示动作产生的新状态的代价。
3. DRL代理通过与环境的交互来训练本地模型，然后使用DDPG来选择IIoT设备节点的最优方案。在DRL算法中，我们使用奖励函数来评估采取行动的效果。
4. DDPG使用一个值函数来确定策略。

## 实验
1. 数据集： MNIST 和 Fashion MNIST（用于数字识别任务）。 MNIST：使用简单的多层感知(MLP)，有两个隐藏层，每层有200个单元，由ReLU函数激活。 Fashion MNIST：由CNN训练，CNN有一个全连接层(512个单元和ReLU激活函数)，一个softmax输出层和两个5×5卷积层。
2. 两种数据划分方法：数据混洗的IID和 non-IID。IID最后分成100台设备分别接收600个数据样本。non-iid，根据数字标签对数据进行分类，300块分成200块，每个客户分配2块。这种设置方法可以探索算法对高度非iid数据的破坏程度。
3. 数据集：CIFAR-10。 将里面的图像用作测试集和训练集。
