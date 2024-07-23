# Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition

[【Paper】](https://arxiv.org/abs/2401.17270) [【Code】](https://github.com/AILab-CVC/YOLO-World) 

# 1.1 Conv2Former 论文解读：
## 1.1.1 背景和动机

以 VGGNet、Inception 系列和 ResNet 系列为代表的 2010-2020 年代的卷积神经网络 (ConvNets) 在多种视觉任务中取得了巨大的进展，它们的共同特点是顺序堆叠多个基本模块 (Basic Building Block)，并采用金字塔结构 (pyramid network architecture)，但是却忽略了显式建模全局上下文信息的重要性。SENet 模块系列模型突破了传统的 CNN 设计思路，将注意力机制引入到 CNN 中以捕获远程依赖，获得了更好的性能。

自从 2020 年以来，视觉 Transformer (ViTs) 进一步促进了视觉识别模型的发展，在 ImageNet 图像分类和下游任务上表现出比最先进的 ConvNets 更好的结果。这是因为与只进行局部建模的卷积操作相比，Transformer 中的自注意力机制能够对全局的成对依赖进行建模，提供了一种更有效的空间信息编码方法。然而，在处理高分辨率图像时，自注意力机制导致的计算成本是相当大的。

为了解决这个问题，一些 2022 年经典的工作试图回答：如何借助卷积操作，打造具有 Transformer 风格的卷积网络视觉基线模型？

比如 ConvNeXt[1]：将标准 ResNet 架构现代化，并使用与 Transformer 相似的设计和训练策略，ConvNeXt 可以比一些 Transformer 表现得更好。

再比如 HorNet[2]：通过建模高阶的相互作用，使得纯卷积模型可以做到像 Transformer 一样的二阶甚至更高的相互作用。

再比如 RepLKNet[3]，SLaK[4]：通过 31×31 或者 51×51 的超大 Kernel 的卷积，使得纯卷积模型可以建模更远的距离。

到目前为止，如何更有效地利用卷积来构建强大的 ConvNet 体系结构仍然是一个热门的研究课题。


## 1.1.2 卷积调制模块
<img width="712" alt="image" src="https://github.com/user-attachments/assets/fda81ee2-7b5e-4754-b518-4ddd36fb6959">

<img width="698" alt="image" src="https://github.com/user-attachments/assets/d2765df3-a85b-409b-99ed-79e435f124b3">

ConvNeXt 表明，将 ConvNets 的核大小从3扩大到7可以提高分类性能。然而，进一步增加 Kernel 的大小几乎不会带来性能上的提升，反而会在没有重新参数化的情况下增加计算负担。但作者认为，使 ConvNeXt 从大于 7×7的 Kernel Size 中获益很少的原因是使用空间卷积的方式。对于 Conv2Former，当 Kernel Size 从 5×5 增加到 21×21 时，可以观察到一致的性能提升。这种现象不仅发生在 Conv2Former-T (82.8→83.4) 上，也发生在参数为80M+ 的 Conv2Former-B (84.1→84.5) 上。考虑到模型效率，默认的 Kernel Size 大小可以设置为 11×11。

![image](https://github.com/user-attachments/assets/f1d34917-827b-49cc-9570-a7fc5c4c7ddf)

## 1.1.3 Conv2Former 整体架构

如下图3所示，与ConvNeXt 和 Swin Transformer 相似，作者的 Conv2Former 也采用了金字塔架构。总共有4个 Stage，每个 Stage 的特征分辨率依次递减。根据模型大小尺寸，一共设计了5个变体：Conv2Former-N，Conv2Former-T， Conv2Former-S， Conv2Former-B，Conv2Former-L。

![image](https://github.com/user-attachments/assets/2020873d-fea3-40be-b34a-150c28b3921e)

当可学习参数数量固定时，如何安排网络的宽度和深度对模型性能有影响。原始的 ResNet-50 将每个 Stage 的块数设置为 (3,4,6,3)。ConvNeXt-T 按照 Swin-T 的模式将 Block 数之比更改为 (3,3,9,3)，并对较大的模型将 Block 数之比更改为 (1,1,9,1)。Conv2Former 的设置如下图4所示。可以观察到，对于一个小模型 (参数小于30M)，更深的网络表现更好。

![image](https://github.com/user-attachments/assets/3b2baeec-1835-4f7d-a364-6024e11d53a8)
