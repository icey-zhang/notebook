# YOLO-World: Real-Time Open-Vocabulary Object Detection

<img width="600" alt="image" src="https://github.com/icey-zhang/notebook/assets/54712081/0382fa34-c3d2-4732-99b8-d50697215c6b">

[【Paper】](https://arxiv.org/abs/2401.17270) [【Code】](https://github.com/AILab-CVC/YOLO-World) [【Data】](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/data.md)

## 摘要
You Only Look Once (YOLO)系列检测器已将自己确立为高效实用的工具。然而，它们对预定义和经过训练的对象类别的依赖限制了它们在开放场景中的适用性。为了解决这一限制，我们引入了 YOLO-World，这是一种创新的方法，通过视觉语言建模和大规模数据集的预训练来增强 YOLO，具有开放词汇检测能力。具体来说，我们提出了一种新的可重新参数化视觉语言路径聚合网络（RepVL-PAN）和区域-文本对比损失，以促进视觉和语言信息之间的交互。我们的方法擅长以高效率的零样本方式检测广泛的对象。在具有挑战性的LVIS数据集上，YOLO-World在V100上实现了35.4 AP，在精度和速度方面都优于许多最先进的方法。此外，微调后的 YOLO-World 在几个下游任务（包括对象检测和开放词汇实例分割）上取得了显着的性能。

## 1.介绍
目标检测是计算机视觉中一个长期而根本的挑战，在图像理解、机器人和自动驾驶汽车等领域有着广泛的应用。大量的工作[16,27,43,45]随着深度神经网络的发展，目标检测取得了重大突破。尽管这些方法取得了成功，但它们仍然有限，因为它们只处理具有固定词汇表的对象检测，例如 COCO [26] 数据集中的 80 个类别。一旦定义了对象类别并标记，经过训练的检测器只能检测这些特定的类别，从而限制了开放场景的能力和适用性。最近的工作[8,13,48,53,58]探索了流行的视觉语言模型[19,39]，通过从语言编码器(如BERT[5])中提取词汇知识来解决开放词汇检测[58]。然而，由于词汇量有限的训练数据稀缺，这些基于蒸馏的方法非常有限，例如包含 48 个基本类别的 OV-COCO [58]。几种方法[24,30,56,57,59]将目标检测训练重新定义为**区域级视觉语言预训练**，并大规模训练开放词汇对象检测器。然而，这些方法在现实场景中仍然难以检测，**这有两个方面：（1）沉重的计算负担和（2）边缘设备的复杂部署。**

先前的工作已经展示了**预训练大型检测器**的良好性能，同时**预训练小型检测器**以**赋予它们开放识别能力**仍未得到探索。

在本文中，我们提出了 YOLO-World，旨在高效开放词汇对象检测，并探索大规模预训练方案，将传统的 YOLO 检测器提升到一个新的开放词汇世界。与以前的方法相比，所提出的 YOLOWorld 在推理速度高且易于部署到下游应用程序中非常有效。具体来说，**YOLO-World 遵循标准的 YOLO 架构 [20]，并利用预训练的 CLIP [39] 文本编码器对输入文本进行编码**。我们进一步提出了可**重新参数化的视觉语言路径聚合网络（RepVL-PAN）来连接文本特征和图像特征以获得更好的视觉语义表示**。在推理过程中，可以删除文本编码器，并将文本嵌入重新参数化为 RepVL-PAN 的权重以实现高效部署。我们通过大规模数据集上的区域-文本对比学习进一步研究了YOLO检测器的开放词汇预训练方案，将检测数据、接地数据和图像-文本数据统一为区域-文本对。具有丰富区域-文本对的预训练 YOLO-World 在大规模词汇检测和训练更多数据方面表现出强大的能力，可以显着提高开放词汇能力。

此外，我们探索了一种提示然后检测范式，以进一步提高现实场景中开放词汇对象检测的效率。如图 2 所示，传统的目标检测器 [16, 20, 23, 41-43, 52] 专注于具有预定义和训练类别的**固定词汇表（闭集）检测**。虽然以前的开放词汇检测器[24,30,56,59]**使用文本编码器对用户的提示进行编码**，以检测对象。值得注意的是，这些方法倾向于使用具有大量主干的大型检测器，例如 Swin-L [32]，以增加开放词汇容量。相比之下，prompt-thendetect 范式（图 2 (c)）首先对用户的提示进行编码以构建离线词汇表，词汇表因不同需求而异。然后，高效的检测器可以动态推断离线词汇表，而无需重新编码提示。对于实际应用，一旦我们训练了检测器，即 YOLO-World，我们可以**预先编码提示或类别以构建离线词汇表**，然后将其无缝集成到检测器中。

<img width="682" alt="image" src="https://github.com/icey-zhang/notebook/assets/54712081/ec9685e9-5190-464f-a7d2-c8c560310941">

> 区别
> | 方法             | 描述                                            | 备注 |
> | ---------------- | ----------------------------------------------- | ----------------------------------------------- |
> | 传统目标检测     | 固定词汇表（闭集）检测                          | / |
> | 开放词汇检测器   | 使用文本编码器对用户的提示进行编码，以检测对象  | 倾向于使用的backbone较大的大型检测器，以增加开放词汇容量 |
> | YOLO-World       | 预先编码提示或类别以构建离线词汇表              | 离线词汇表是？高效的检测器可以动态推断离线词汇表，而无需重新编码提示 |

> **“离线词汇表”**是指在目标检测过程中预先生成和存储的一组类别或对象名称的集合。这些类别或对象名称由用户提供的提示或文本描述编码而来。在实际检测过程中，系统使用这个预先生成的词汇表进行检测，而不是在每次检测时重新编码用户的提示。


> 我们的主要贡献可以概括为三个方面： 
>
> • 我们介绍了 YOLO-World，这是一种尖端开放词汇对象检测器，在实际应用中具有高效率。
>
> • 我们提出了一种可重新参数化的视觉语言PAN来连接视觉和语言特征和YOLO-World的开放词汇区域-文本对比预训练方案。
>
> • 在大规模数据集上预训练的 YOLO-World 表现出强大的零样本性能，并在 LVIS 上以 52.0 FPS 实现了 35.4 AP。预训练的 YOLO-World 可以轻松适应下游任务，例如开放词汇实例分割和参考对象检测。此外，YOLO-World 的预训练权重和代码将被开源，以促进更实用的应用。


## 2.相关工作
### 开放词汇对象检测 (OVD)
开放词汇对象检测 (OVD) [58] 已成为现代目标检测的新趋势，旨在检测预定义类别之外的对象。早期的工作[13]通过在基类上训练检测器并评估新的(未知)类，遵循标准的OVD设置[58]。然而，这种开放词汇设置可以评估检测器检测和识别新对象的能力，对于开放场景仍然受到限制，并且由于在有限的数据集和词汇上进行训练，缺乏对其他领域的泛化能力。 [58] 已成为现代目标检测的新趋势，旨在检测预定义类别之外的对象。早期的工作[13]通过在基类上训练检测器并评估新的(未知)类，遵循标准的OVD设置[58]。然而，这种开放词汇设置可以评估检测器检测和识别新对象的能力，对于开放场景仍然受到限制，并且由于在有限的数据集和词汇上进行训练，缺乏对其他领域的泛化能力。

受视觉语言预训练[19,39]启发，最近的工作[8,22,53,62,63]将开放词汇对象检测表述为图像-文本匹配，并利用大规模图像-文本数据大规模增加训练词汇。**OWLViT** [35, 36] 使用检测和接地数据集微调简单的视觉转换器 [7]，并构建具有有希望的性能的简单开放词汇检测器。**GLIP** [24] 提出了一种基于短语Grounding的开放词汇检测框架，并在零样本设置中进行评估。Grounding DINO[30]将接地的预训练[24]合并到具有跨模态融合的检测变压器[60]中。几种方法[25,56,57,59]通过区域-文本匹配和大规模图像-文本对的预训练检测器统一检测数据集和图像-文本数据集，取得了良好的性能和泛化能力。然而，这**些方法通常使用像ATSS[61]或DINO[60]这样的重检测器，Swin-L[32]作为骨干，导致计算需求高和部署挑战**。相比之下，我们提出了 YOLO-World，旨在通过实时推理和更容易的下游应用程序部署进行有效的开放词汇对象检测。与 <font color=Blue>**ZSD-YOLO** [54]</font>不同，ZSD-YOLO [54] 还通过语言模型对齐探索了带有 YOLO 的开放词汇检测 [58]，YOLO-World 引入了一种新颖的 YOLO 框架，具有**有效的预训练策略**，增强了开放词汇性能和泛化。

<img width="833" alt="image" src="https://github.com/icey-zhang/notebook/assets/54712081/9f0fe4eb-a1d7-4144-8f9c-c3e59c08a8e3">


## 3.方法
### 方法部分详细翻译(来自Chatgpt4o)

#### 3.1 预训练公式：区域-文本对

传统的目标检测方法，包括 YOLO 系列 [20]，是用实例注释 $`\Omega = \{B_i, c_i\}_{i=1}^N`$ 进行训练的，其中包括边界框 $`\{B_i\}`$ 和类别标签 $`\{c_i\}`$。本文将实例注释重新定义为区域-文本对 $`\Omega = \{B_i, t_i\}_{i=1}^N`$，其中 $`t_i`$ 是对应于区域 $`B_i`$ 的文本。具体来说，文本 $`t_i`$ 可以是类别名称、名词短语或物体描述。此外，YOLO-World 采用图像 $`I`$ 和文本 $`T`$（一组名词）作为输入，并输出预测框 $`\{B_k\}`$ 及相应的物体嵌入 $`\{e_k\}`$（$`e_k \in \mathbb{R}^D`$）。

> 从（边界框+类别标签）--> 到（边界框+文本【类别名称，名次短语，物体描述】）
> - 输入：图像 $`I`$ 和文本 $`T`$（一组名词）
> - 输出：预测框 $`\{B_k\}`$ 及相应的物体嵌入 $`\{e_k\}`$（$`e_k \in \mathbb{R}^D`$）

#### 3.2 模型架构

本文提出的 YOLO-World 的总体架构如图 3 所示，包括一个 **YOLO 检测器**、一个**文本编码器**和一个**可重新参数化的视觉-语言路径聚合网络（RepVL-PAN）**。给定输入文本，YOLO-World 中的文本编码器将文本编码成文本嵌入。YOLO 检测器中的图像编码器从输入图像中提取多尺度特征。然后我们利用 **RepVL-PAN 通过图像特征和文本嵌入之间的跨模态融合来增强文本和图像的表示**。


> 1. **YOLO 检测器**
>    - YOLO检测器用于目标检测任务，在图像中识别不同的对象。
>
> 2. **文本编码器**
>    - 文本编码器将文本信息转换为向量表示，以便在模型中进行处理。
>
> 3. **可重新参数化的视觉-语言路径聚合网络（RepVL-PAN）**
>    - RepVL-PAN是一种结合了视觉和语言信息的神经网络架构，用于在视觉和语言之间进行信息传递和聚合。


##### YOLO 检测器

YOLO-World 主要基于 YOLOv8 [20] 开发，包含一个 **Darknet 骨干网络** [20, 43] 作为**图像编码器**，一个**路径聚合网络（PAN）**用于**多尺度特征金字塔**，以及一个**用于边界框回归和物体嵌入**的**检测head**。

> Base | YOLOv8
> - Darknet 骨干网络 ｜ 图像编码器
> - 路径聚合网络PAN ｜ 多尺度特征金字塔
> - 检测头 ｜ 边界框回归和物体嵌入

##### 文本编码器

给定文本 $`T`$，我们采用由 **CLIP [39] 预训练的 Transformer 文本编码器**来提取相应的文本嵌入 $`W = \text{TextEncoder}(T) \in \mathbb{R}^{C \times D}`$，其中 $`C`$ 是名词的数量，$`D`$ 是嵌入维度。与仅文本的语言编码器 [5] 相比，CLIP 文本编码器在连接视觉对象和文本方面具有更好的视觉语义能力。当输入文本是标题或引用表达时，我们采用简单的 **[【n-gram 算法】](https://blog.csdn.net/qq_41667743/article/details/129453006)来提取名词短语**，然后将其输入到文本编码器中。

> 预训练的 Transformer 文本编码器

##### 文本对比头

跟随之前的工作 [20]，我们采用解耦头部，通过两个 3×3 的卷积来回归边界框 $`\{b_k\}_{k=1}^K`$ 和物体嵌入 $`\{e_k\}_{k=1}^K`$，其中 $`K`$ 表示物体的数量。我们提出一个文本对比头，以获得物体-文本相似性 $`s_{k,j}`$，公式如下：

$$ s_{k,j} = \alpha \cdot \text{L2-Norm}(e_k) \cdot \text{L2-Norm}(w_j)^\top + \beta $$

其中 $`\text{L2-Norm}(\cdot)`$ 是 L2 归一化，$`w_j \in W`$ 是第 $`j`$ 个文本嵌入。此外，我们添加了具有可学习缩放因子 $`\alpha`$ 和偏移因子 $`\beta`$ 的仿射变换。L2 归一化和仿射变换对于稳定区域-文本训练非常重要。

> 类似于CLIP里面的相似度矩阵

##### 在线词汇训练

在训练过程中，我们为**每个包含 4 张图像的马赛克样本**构建一个**在线词汇表 $`T`$**。具体来说，我们对马赛克图像中涉及的所有正向名词进行采样，并从相应的数据集中随机采样一些负向名词。**每个马赛克样本的词汇最多包含 $`M`$ 个名词，默认设置 $`M`$ 为 80**。

##### 离线词汇推理

在推理阶段，我们提出了一种带有离线词汇的提示-然后-检测策略，以进一步提高效率。如图 3 所示，**用户可以定义一系列自定义提示，包括标题或类别**。然后我们利用文本编码器对这些提示进行编码，获得离线词汇嵌入。离线词汇避免了每次输入的计算，并提供了根据需要调整词汇的灵活性。

<img width="671" alt="image" src="https://github.com/icey-zhang/notebook/assets/54712081/af4430ec-72ca-4550-be0e-10412bd09609">


#### 3.3 可重新参数化的视觉-语言路径聚合网络

图 4 显示了提出的 RepVL-PAN 的结构，该结构遵循 [20, 29] 中的自顶向下和自底向上路径，建立多尺度图像特征金字塔 $`\{P_3, P_4, P_5\}`$。此外，我们提出了文本引导的 **CSPLayer（T-CSPLayer）**和**图像池化注意力（I-Pooling Attention）**，以进一步增强图像特征和文本特征之间的交互，这可以提高开放词汇能力的视觉语义表示。在推理过程中，**离线词汇嵌入可以重新参数化为卷积层或线性层的权重，以进行部署**。

<img width="402" alt="image" src="https://github.com/icey-zhang/notebook/assets/54712081/dca0b692-599f-4161-b227-2dbaddbeadad">


##### 文本引导的 CSPLayer

如图 4 所示，跨阶段部分层（CSPLayer）在自顶向下或自底向上融合后被利用。我们通过将文本指导纳入多尺度图像特征，扩展了 [20] 的 CSPLayer（也称为 C2f），形成文本引导的 CSPLayer。具体来说，给定文本嵌入 $`W`$ 和图像特征 $`X_l \in \mathbb{R}^{H \times W \times D}`$（$`l \in \{3, 4, 5\}`$），我们在最后一个黑暗瓶颈块之后采用最大-sigmoid 注意力，将文本特征聚合到图像特征中，公式如下：

$$ X'_l = X_l \cdot \delta \left( \max_{j \in \{1..C\}} (X_l W_j^\top) \right)^\top $$

其中更新后的 $`X'_l`$ 与跨阶段特征连接作为输出。$`\delta`$ 表示 sigmoid 函数。

##### 图像池化注意力

为了增强具有图像感知信息的文本嵌入，我们通过提出图像池化注意力来聚合图像特征以更新文本嵌入。与直接使用图像特征上的交叉注意力不同，我们在多尺度特征上采用最大池化来获得 3×3 区域，结果为 27 个补丁令牌 $`X̃ \in \mathbb{R}^{27 \times D}`$。然后我们通过以下公式更新文本嵌入：

$$ W' = W + \text{MultiHead-Attention}(W, X̃, X̃) $$

#### 3.4 预训练方案

本节介绍了在大规模检测、定位和图像-文本数据集上预训练 YOLO-World 的训练方案。

##### 使用区域-文本对比损失进行学习

给定马赛克样本 $`I`$ 和文本 $`T`$，YOLO-World 输出 $`K`$ 个物体预测 $`\{B_k, s_k\}_{k=1}^K`$ 以及注释 $`\Omega = \{B_i, t_i\}_i^N`$。我们遵循 [20] 的方法，利用任务对齐标签分配 [9] 将预测与真实注释匹配，并将每个正向预测分配一个文本索引作为分类标签。基于此词汇，我们通过交叉熵在物体-文本（区域-文本）相似性和物体-文本分配之间构建区域-文本对比损失 $`L_{\text{con}}`$。此外，我们采用 IoU 损失和分布式焦点损失进行边界框回归，整体训练损失定义如下：

$$ L(I) = L_{\text{con}} + \lambda_I \cdot (L_{\text{iou}} + L_{\text{dfl}}) $$

其中 $`\lambda_I`$ 是一个指示因子，当输入图像 $`I`$ 来自检测或定位数据时，设置为 1；当输入图像来自图像-文本数据时，设置为 0。考虑到图像-文本数据集包含噪声框，我们只对具有准确边界框的样本计算回归损失。

##### 使用图像-文本数据进行伪标签生成

与直接使用图像-文本对进行预训练不同，我们提出了一种自动标注方法来生成区域-文本对。具体来说，该标注方法包含三个步骤：（1）提取名词短语：我们首先利用 n-gram 算法从文本中提取名词短语；（2）伪标签生成：我们采用预训练的开放词汇检测器（如 GLIP [24]）为每张图像生成伪边界框及其对应的名词短语，从而提供粗略的区域-文本对；（3）过滤：我们利用预训练的 CLIP [39] 评估图像-文本对和区域-文本对的相关性，并过滤掉低相关性的伪标注和图像。我们进一步通过非极大值抑制（NMS）等方法过滤冗余的边界框。详细方法请参考附录。通过上述方法，**我们从 CC3M [47] 数据集中采样并标注了 246k 张图像，共生成了 821k 个伪标注。**

#### 3.4.1 训练方案

在这一部分中，我们介绍了 YOLO-World 在大规模检测、定位和图像-文本数据集上预训练的训练方案。

###### 使用区域-文本对比损失进行学习

给定马赛克样本 $`I`$ 和文本 $`T`$，YOLO-World 输出 $`K`$ 个**物体预测** $`\{B_k, s_k\}_{k=1}^K`$ 以及**annotations** $`\Omega = \{B_i, t_i\}_i^N`$。我们遵循 [20] 的方法，利用任务对齐标签分配 [9] 将预测与真实注释匹配，并将每个正向预测分配一个文本索引作为分类标签。基于此词汇，我们通过交叉熵在物体-文本（区域-文本）相似性和物体-文本分配之间构建区域-文本对比损失 $`L_{\text{con}}`$。此外，我们采用 IoU 损失和分布式焦点损失进行边界框回归，整体训练损失定义如下：

$$ L(I) = L_{\text{con}} + \lambda_I \cdot (L_{\text{iou}} + L_{\text{dfl}}) $$

其中 $`\lambda_I`$ 是一个指示因子，当输入图像 $`I`$ 来自检测或定位数据时，设置为 1；当输入图像来自图像-文本数据时，设置为 0。考虑到图像-文本数据集包含噪声框，我们只对具有准确边界框的样本计算回归损失。

###### 使用图像-文本数据进行伪标签生成

与直接使用图像-文本对进行预训练不同，我们提出了一种自动标注方法来生成区域-文本对。具体来说，该标注方法包含三个步骤：（1）提取名词短语：我们首先利用 **n-gram 算法**从文本中**提取名词短语**；（2）伪标签生成：我们采用预训练的**开放词汇检测器（如 GLIP [24]）**为**每张图像生成伪边界框及其对应的名词短语**，从而提供粗略的区域-文本对；（3）过滤：我们利用**预训练的 CLIP [39] 评估图像-文本对和区域-文本对的相关性**，并过滤掉低相关性的伪标注和图像。我们进一步通过非极大值抑制（NMS）等方法过滤冗余的边界框。详细方法请参考附录。**通过上述方法，我们从 CC3M [47] 数据集中采样并标注了 246k 张图像，共生成了 821k 个伪标注**。


<img width="378" alt="image" src="https://github.com/icey-zhang/notebook/assets/54712081/74eac72b-ec32-4ef4-824b-d67579933202">

> "grounding"（对齐）通常指的是将图像中的视觉元素与文本描述或标签对齐

## 4.实验
### 实验部分详细总结(来自Chatgpt4o)

在本部分中，我们通过在大规模数据集上预训练 YOLO-World 并在 LVIS 和 COCO 基准上进行零样本和微调评估，展示了 YOLO-World 的有效性。

#### 4.1 实现细节

YOLO-World 是基于 MMYOLO 和 MMDetection 工具箱开发的。按照 [20]，我们为不同的延迟需求提供了三种变体，即小型（S）、中型（M）和大型（L）。我们采用开源的 CLIP 文本编码器，并使用预训练权重来编码输入文本。除非特别说明，否则所有模型的推理速度均在一台 NVIDIA V100 GPU 上进行测量，无额外加速机制，如 FP16 或 TensorRT。

#### 4.2 预训练

##### 实验设置

在预训练阶段，我们采用 AdamW 优化器，初始学习率为 0.002，权重衰减为 0.05。YOLO-World 在 32 块 NVIDIA V100 GPU 上预训练了 100 个 epoch，总批量为 512。在预训练过程中，我们遵循之前的工作，采用颜色增强、随机仿射、随机翻转和 4 张图像的马赛克进行数据增强。预训练期间文本编码器被冻结。

##### 预训练数据

为了预训练 YOLO-World，我们主要采用检测或定位数据集，包括 **Objects365 (V1)、GQA 和 Flickr30k**。根据 [24]，我们排除了 GoldG 中 COCO 数据集的图像（GQA 和 Flickr30k）。检测数据集的注释包含边界框和类别或名词短语。此外，我们还使用**图像-文本对，即通过伪标签方法标注的 CC3M 数据集**。

> **O365（Objects365)** 是一个大规模的目标检测数据集，专门设计用于训练和评估计算机视觉模型，特别是在目标检测任务中。它包含了丰富的类别和大量的图像样本，旨在推动目标检测技术的发展和应用。
> **GQA（Graph Question Answering** 是一个用于视觉问答（Visual Question Answering, VQA）的数据集，专注于基于图像的复杂推理和理解任务。GQA 数据集包含超过22万个图像，约1400万个问题。每个问题都与图像中的对象和关系相关联，提供了丰富的训练和评估资源。
> **Flickr30k** 是一个广泛用于计算机视觉和自然语言处理领域的图像-文本数据集，特别用于图像描述生成（image captioning）和视觉问答（visual question answering, VQA）等任务。该数据集由30000张图片及其对应的文本描述组成，最初是从Flickr图片分享网站上收集的。每张图片都配有5个不同的文本描述，这些描述详细地说明了图像中的场景、对象及其相互关系。

#### 4.3 零样本评估

在预训练后，我们直接在 LVIS 数据集上对 YOLO-World 进行零样本评估。LVIS 数据集包含 1203 个对象类别，远多于预训练检测数据集的类别，可以衡量大词汇检测的性能。我们主要在 LVIS minival 上进行评估，并报告 Fixed AP 以进行比较。最大预测数量设置为 1000。

#### 4.4 消融实验

我们进行了广泛的消融研究，从两个主要方面分析了 YOLO-World，即预训练和架构。

##### 预训练数据

与基于 Objects365 训练的基线相比，增加 GQA 可以显著提高性能，在 LVIS 上增加了 8.4 AP。这一改进可以归因于 GQA 数据集提供的更丰富的文本信息，这可以增强模型识别大词汇对象的能力。增加部分 CC3M 样本（占全部数据集的 8%）可以进一步带来 0.5 AP 增益，并在稀有对象上带来 1.3 AP 增益。

##### RepVL-PAN 的消融实验

我们通过在两个设置下验证了提出的 RepVL-PAN 的有效性：（1）基于 O365 预训练和（2）基于 O365 和 GQA 预训练。与仅包含类别注释的 O365 相比，GQA 包含丰富的文本，特别是名词短语。实验表明，提出的 RepVL-PAN 提高了 LVIS 上的基线 YOLOv8-PAN，尤其是在 LVIS 的稀有类别（APr）方面显著改善。

#### 4.5 微调 YOLO-World

在这一部分中，我们进一步在 COCO 和 LVIS 数据集上微调 YOLO-World，以展示预训练的有效性。

##### 实验设置

我们使用预训练权重初始化 YOLO-World 进行微调。所有模型在 AdamW 优化器下微调 80 个 epoch，初始学习率设置为 0.0002。此外，我们以 0.01 的学习率微调 CLIP 文本编码器。对于 LVIS 数据集，我们遵循之前的工作，在 LVIS-base（常见和频繁类别）上微调 YOLO-World，并在 LVIS-novel（稀有类别）上进行评估。

##### COCO 目标检测

我们将预训练的 YOLO-World 与之前的 YOLO 检测器进行了比较。在 COCO 数据集上微调 YOLO-World 时，我们移除了提出的 RepVL-PAN 以进一步加速，考虑到 COCO 数据集的词汇量较小。实验结果显示，我们的方法在 COCO 数据集上表现出色，微调后的 YOLO-World 在 COCO train2017 上展示了较高的性能。

##### LVIS 目标检测

我们在标准 LVIS 数据集上评估了微调 YOLO-World 的性能。与在完整 LVIS 数据集上训练的 YOLOv8s 相比，YOLO-World 显著改进，尤其是在较大模型上。例如，YOLO-World-L 超过 YOLOv8-L 7.2 AP 和 10.2 APr。

#### 4.6 开放词汇实例分割

在本部分中，我们进一步微调 YOLO-World 在开放词汇实例分割（OVIS）下的表现。与之前的方法不同，我们直接在包含掩码注释的子集上微调 YOLO-World，并在大词汇设置下评估分割性能。

##### 实验结果

实验结果显示，YOLO-World 在 COCO 到 LVIS 设置和 LVIS-base 到 LVIS 设置下均表现出色。微调分割头部或所有模块的策略均能显著提高性能。

#### 4.7 可视化

我们提供了预训练的 YOLO-World-L 在三种设置下的可视化结果：（a）使用 LVIS 类别进行零样本推理；（b）使用自定义提示进行细粒度检测；（c）引用检测。可视化结果展示了 YOLO-World 在开放词汇场景下的强大泛化能力以及引用能力。

## 5. 结论

我们提出了 YOLO-World，这是一种先进的实时开放词汇检测器，旨在提高现实世界应用中的效率和开放词汇能力。本文通过将流行的 YOLO 重新设计为一种视觉-语言 YOLO 架构进行开放词汇预训练和检测，提出了连接视觉和语言信息的 RepVL-PAN，并可以重新参数化以便高效部署。实验结果表明，YOLO-World 在速度和开放词汇性能方面具有明显优势，并证明了在小模型上进行视觉-语言预训练的有效性，这对未来研究具有启发性。我们希望 YOLO-World 能作为解决现实世界开放词汇检测的新基准。


## 复现教程

[【Readme】](https://docs.ultralytics.com/zh/models/yolo-world/#benefits-of-saving-with-custom-vocabulary)

[【Bug】](https://blog.csdn.net/ITdaka/article/details/138863017)

