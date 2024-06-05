GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer

<img width="664" alt="image" src="https://github.com/icey-zhang/notebook/assets/54712081/b4372f5f-7e8a-48dc-9f50-80ba946b7521">

# 方法

## 基于交换的融合
基于交换的方法，如TokenFusion和CEN，旨在动态检测和替换单模态变压器中的无用token或通道，使用来自其他模态的特征进行替换。具体来说，TokenFusion的核心功能是修剪每个模态中的token，并用从其他模态投影和聚合的相应token替换它们。这个交换过程由网络中的得分预测器引导，该预测器计算与多模态输入共享维度的掩码，这些掩码通过与预定义阈值进行比较来选择要替换的token。

公式如下：

$$
X1[i] = X1[i] \cdot \mathbb{I}(s(X1[i]) \geq \theta) + X2[i] \cdot \mathbb{I}(s(X1[i]) < \theta)
$$

$$
X2[i] = X2[i] \cdot \mathbb{I}(s(X2[i]) \geq \theta) + X1[i] \cdot \mathbb{I}(s(X2[i]) < \theta)
$$

其中，$`X1[i]`$表示输入$`X1`$的第$`i`$个token，$`\mathbb{I}`$ 是断言下标条件的指示符，输出一个掩码张量，参数 $\theta$ 是一个小阈值，取值为 0.02，操作符 $\cdot$ 表示逐元素相乘。

## 基于交叉注意力的融合
交叉注意力机制通过处理来自多模态的输入，生成增强的多模态信息输出。具体公式如下：

$$
Y1 = \text{Attention}(X1 W_Q, X2 W_K, X2 W_V) + X1
$$

$$
Y2 = \text{Attention}(X2 W_Q, X1 W_K, X1 W_V) + X2
$$

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V
$$

其中，$`X1`$ 和 $`X2`$ 分别是来自两种模态的输入token，$`Q, K, V`$ 分别表示查询、键和值矩阵，$`d`$ 是缩放因子。

## GeminiFusion: 像素级融合模块
为了在保持高效的同时利用交叉注意力机制的优点，提出了一种像素级融合模块——GeminiFusion。

- **原理**：不是所有的patch在融合过程中都同等重要。较不显著的patch可以被来自另一模态的空间对应patch高效替代，这意味着不需要所有patch之间的详尽交互。GeminiFusion模块优先考虑来自不同模态的空间共址patch之间的交互，从而优化交叉注意力机制。

公式如下：

$$
Y1[i] = \text{Attention}(X1[i] W_Q, X2[i] W_K, X2[i] W_V) + X1[i]
$$

$$
Y2[i] = \text{Attention}(X2[i] W_Q, X1[i] W_K, X1[i] W_V) + X2[i]
$$

其中，$`i`$ 表示范围 $`d`$ 内的像素索引。

- **自适应噪声**：在自注意力机制中，**我们增加了层自适应噪声来动态平衡自注意力和跨模态注意力**，确保 softmax 操作的适当功能。

公式如下：

$$
Q1 = X1[i] W_Q
$$

$$
K1 = [(Noise_{KL} + X1[i]) W_K, X1[i] \phi(X1[i], X2[i]) W_K]
$$

$$
V1 = [(Noise_{VL} + X1[i]) W_V, X2[i] W_V]
$$

$$
Y1[i] = \text{Attention}(Q1, K1, V1) + X1[i]
$$

$$
Q2 = X2[i] W_Q
$$

$$
K2 = [(Noise_{KL} + X2[i]) W_K, X2[i] \phi(X2[i], X1[i]) W_K]
$$

$$
V2 = [(Noise_{VL} + X2[i]) W_V, X1[i] W_V]
$$

$$
Y2[i] = \text{Attention}(Q2, K2, V2) + X2[i]
$$

其中，$`\phi(·)`$ 表示关系鉴别模块，用于评估模态之间的差异。

## 总体架构
GeminiFusion模型采用编码器-解码器架构，编码器包括四级结构，类似于SegFormer，用于提取分层特征。每个模态都通过GeminiFusion模块提炼特征，并在解码器中综合这些特征以生成分割预测。解码器使用基于MLP的设计，以增强预测的泛化能力。

通过这种方法，GeminiFusion能够高效地处理多模态融合任务，在保留模态内特征的同时，有效利用跨模态特征，实现了出色的性能。

## 关键技术贡献
1. 通过像素级的交叉注意力机制实现了高效的多模态融合。
2. 自适应噪声的引入平衡了自注意力和跨模态注意力，避免了信息损失。
3. 提出了一种轻量级的关系鉴别模块，增强了特征生成过程的精确性。

这些创新使GeminiFusion在多种多模态任务中表现出色，包括语义分割、图像到图像翻译和3D物体检测，为未来的多模态融合研究和应用提供了坚实的基础。



