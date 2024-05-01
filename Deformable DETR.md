DETR 《End-to-end object detection with transformers》，ECCV 2020

Deformable DETR 《Deformable DETR: Deformable Transformers for End-to-End Object Detection》，ICLR 2021

Deformable Attention 《Vision Transformer with Deformable Attention》,CVPR 2022

Deformable Attention（可变形注意力）首先在2020年10月初商汤研究院的[《Deformable DETR: Deformable Transformers for End-to-End Object Detection》](https://paperswithcode.com/paper/deformable-detr-deformable-transformers-for-1)论文中提出，在2022CVPR中[《Vision Transformer with Deformable Attention》](https://openaccess.thecvf.com/content/CVPR2022/html/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.html)提出应用了**Deformable Attention（可变形自注意力）机制**的通用视觉Transformer骨干网络DAT（Deformable Attention Transformer），并且在多个数据集上效果优于swin transformer。


目录
- DERT原理分析
- Deformable Convolution原理分析
- Deformable DETR 原理分析
- Deformable Attention Module
- Multi-scale Deformable Attention Module

## DERT原理分析

前言
贡献/特点：

1. **端到端：去除NMS和anchor**，没有那么多的超参，**计算量也大大减少**，整个网络变得很简单；
2. 基于Transformer：首次将**Transformer引入到目标检测任务当中**；
3. 提出一种全新的基于集合的损失函数：通过二分图匹配的方法强制模型输出一组独一无二的预测框，**每个物体只会产生一个预测框**，这样就将目标检测问题直接转换为集合预测的问题，所以才不用nms，达到端到端的效果；
4. 而且在**decoder输入一组可学习的object query和encoder输出的全局上下文特征**，直接以并行方式强制输出最终的100个预测框，替代了anchor；
5. 缺点：对大物体的检测效果很好，但是**对小物体的检测效果不好；训练起来比较慢；**
6. 优点：在COCO数据集上速度和精度和Faster RCNN差不多；可以扩展到很多任务中，比如分割、追踪、多模态等；

### 一、整体架构
![image](https://github.com/icey-zhang/notebook/assets/54712081/e910aaa4-66a2-43d6-b1e6-d227e6c74666)

1. 图片输入，首先经过一个**CNN网络提取图片的局部特征**；
2. 再把特征拉直，**输入Transformer Encoder中**，进一步学习这个特征的全局信息。经过Encoder后就可以计算出每一个点或者每一个特征和这个图片的其他特征的相关性；
3. 再把Encoder的输出送入Decoder中，**并且这里还要输入Object Query，限制解码出100个框**，这一步作用就是生成100个预测框；
4. 预测出的100个框和gt框，通过<font color=Blue>**二分图匹配**</font>的方式，确定其中哪些预测框是有物体的，哪些是没有物体的（背景），再把**有物体的框和gt框一起计算分类损失和回归损失**；推理的时候更简单，**直接对decoder中生成的100个**预测框**设置一个置信度阈值(0.7)**，大于的保留，小于的抑制；

### 二、基于集合预测的损失函数
#### 2.1、二分图匹配确定有效预测框
预测得到N（100）个预测框，gt为M个框，通常N>M，那么怎么计算损失呢?

这里呢，就先对这100个预测框和gt框进行一个二分图的匹配，先确定每个gt对应的是哪个预测框，最终再计算M个预测框和M个gt框的总损失。

其实很简单，假设现在有一个矩阵，**横坐标就是我们预测的100个预测框，纵坐标就是gt框**，再分别计算每个预测框和其他所有gt框的cost，这样就构成了一个cost matrix，再确定把如何把所有gt框分配给对应的预测框，才能使得最终的总cost最小。

![image](https://github.com/icey-zhang/notebook/assets/54712081/004217b2-d39f-465a-96d2-5222f299e5a0)

这里计算的方法就是很经典的匈牙利算法，通常是调用scipy包中的linear_sum_assignment函数来完成。这个函数的输入就是cost matrix，输出一组行索引和一个对应的列索引，给出最佳分配。

<font color=Blue>**匈牙利算法**</font>通常用来解决二分图匹配问题，具体原理可以看这里： [二分图匈牙利算法的理解和代码](https://www.bilibili.com/video/BV1FZ4y157Te/?spm_id_from=333.337.search-card.all.click&vd_source=5f6bbc1038b075757cb446f800f3cd56) 和 [算法学习笔记(5)：匈牙利算法](https://zhuanlan.zhihu.com/p/96229700)

所以通过以上的步骤，就确定了最终100个预测框中哪些预测框会作为有效预测框，哪些预测框会称为背景。再将有效预测框和gt框计算最终损失（有效预测框个数等于gt框个数）。

#### 2.2、损失函数
损失函数：分类损失+回归损失
![image](https://github.com/icey-zhang/notebook/assets/54712081/b2c1bfca-0210-41bd-b9d1-8a0395b06201)

分类损失：交叉熵损失，去掉log

回归损失：GIOU Loss + L1 Loss

### 三、前向推理
![image](https://github.com/icey-zhang/notebook/assets/54712081/7fea10bf-ba2c-4724-b535-26f6ef3610c8)

DETR前向传播流程：

1. 假设输入图片：3x800x1066；
2. 输入CNN网络(ResNet50)中，走到Conv5，此时对原图片下采样32倍，输出2048x25x34；
3. 经过一个1x1卷积降为，输出256x25x34；
4. 生成位置编码256x25x34，再和前面CNN输出的特征相加，输出256x25x34的特征；
5. 再把特征拉直，变成850x256，输入transformer encoder中；
6. 经过6个encoder模块，进行全局建模，输入同样850x256的特征；
7. 生成一个可学习的object queries（positional embedding）100x256；
8. 将encode输出的全局特征850x256和object queries 100x256一起输入6层decoder中，反复的做自注意力操作，最后得到一个100x256的特征；（细节：**这里每个decoder都会做一次object query的自注意力操作**，第一个decoder可以不做，这主要是为了移除冗余框；为了让模型训练的更快更稳定，所以在Decoder后面加了很多的auxiliary loss，不光在最后一层decoder中计算loss，在之前的decoder中也计算loss）
9. 最后再接上两个feed forward network预测头（全连接层），一个FFN做物体类别的预测（类别个数），另一个FFN做box预测（4 xywh）；
10. 再用这100个预测框和gt框（N个）通过匈牙利算法做最优匹配，找到最终N个有效的预测框，其他的（100-N）框当作背景，舍去；
11. 再用这N个预测框和N个GT框计算损失函数（交叉熵损失，去掉log + GIOU Loss + L1 Loss），梯度回传；

### 四、掉包版代码
论文原文给出的掉包版代码，mAP好像有40，虽然比源码低了2个点，但是代码很简单，只有40多行，方便我们了解整个detr的网络结构：


```python
import torch
from torch import nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        # backbone = resnet50 除掉average pool和fc层  只保留conv1 - conv5_x
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        # 1x1卷积降维 2048->256
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        # 6层encoder + 6层decoder    hidden_dim=256  nheads多头注意力机制 8头   num_encoder_layers=num_decoder_layers=6
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        # 分类头
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # 回归头
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        # 位置编码  encoder输入
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        # query pos编码  decoder输入
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))


    def forward(self, inputs):
        x = self.backbone(inputs)    # [1,3,800,1066] -> [1,2048,25,34]
        h = self.conv(x)             # [1,2048,25,34] -> [1,256,25,34]
        H, W = h.shape[-2:]          # H=25  W=34
        # pos = [850,1,256]  self.col_embed = [50,128]  self.row_embed[:H]=[50,128]
        pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                        self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                        ], dim=-1).flatten(0, 1).unsqueeze(1)
        # encoder输入  decoder输入
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()


detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
detr.eval()
inputs = torch.randn(1, 3, 800, 1066)
logits, bboxes = detr(inputs)
print(logits.shape)   # torch.Size([100, 1, 92])
print(bboxes.shape)   # torch.Size([100, 1, 4])

```
### 五、一些问题
1、为什么ViT只有Encoder，而DETR要用Encoder+Decoder？（从论文实验部分得出结论）
Encoder：**Encoder自注意力主要进行全局建模，学习全局的特征，通过这一步其实已经基本可以把图片中的各个物体尽可能的分开**；

Decoder：这个时候再使用Decoder自注意力，再做目标检测和分割任务，**模型就可以进一步把物体的边界的极值点区域进行一个更进一步精确的划分，让边缘的识别更加精确；**

2、object query有什么用？
**object query是用来替换anchor的，通过引入可学习的object query，可以让模型自动的去学习图片当中哪些区域是可能有物体的**，最终通过object query可以找到100个这种可能有物体的区域。再后面通过二分图匹配的方式找到100个预测框中有效的预测框，进而计算损失即可。

所以说object query就起到了替换anchor的作用，以可学习的方式找到可能有物体的区域，而不会因为使用anchor而造成大量的冗余框。


## Deformable Convolution原理分析
**Deformable Convolution** 将固定形状的卷积过程改造成了能适应物体形状的可变的卷积过程，从而使结构适应物体形变的能力更强。

传统的CNN卷积核是**固定大小的，只能在固定为位置对固定输入特征进行采样**， 为了解决这个问题，研究人员提出了两种解决思路：

- **使用大量的数据进行训练。** 比如用ImageNet数据集，再在其基础上做翻转等变化来扩展数据集，通俗地说就是通过穷举的方法使模型能够适应各种形状的物体，这种方法收敛较慢而且要设计复杂的网络结构才能达到理想的结果。
- **设计一些特殊的算法来适应形变.** 比如SIFT，目标检测时用滑动窗口法来适应目标在不同位置上的分类也属于这类。

对第一种方法，如果用训练中**没有遇到过的新形状物体 (但同属于一类)来做测试**，由于新形状没有训练过，会造成测试不准确，而且靠数据集来适应形变的**训练过程太耗时** ，网络结构也必须设计的很复杂。

对于第二种方法，如果**物体的形状极其复杂** ，要设计出能适应这种复杂结构的算法就更困难了。

为解决该问题，研究人员提出了 **Deformable Convolution** 方法，它对感受野上的**每一个点加一个偏移量** ，偏移的大小是通过学习得到的 ，**偏移后感受野不再是个正方形，而是和物体的实际形状相匹配**。这么做的好处就是无论物体怎么形变，**卷积的区域始终覆盖在物体形状的周围。**


下图为Deformable Convolution的示意图，a 为原始感受野范围，b ~ c 是对感受野上的添加偏移量后的感受野范围，可以看到叠加偏移量的过程可以模拟出**目标移动、尺寸缩放、旋转等各种形变**

 ![image](https://github.com/icey-zhang/notebook/assets/54712081/8acb7810-b2d3-49b3-a11e-4a882b22e233)

完整的可变性卷积如下图所示，注意上面的卷积用于输出偏移量，该偏移量的长宽和输入特征图的长宽一直，维度是输入的两倍 ， 因为同时输出了x和y方向的偏移量。

![image](https://github.com/icey-zhang/notebook/assets/54712081/7e3d34ef-3b22-4864-a2d7-2411295e09f7)

![image](https://github.com/icey-zhang/notebook/assets/54712081/b34c70e8-a357-499c-af6b-ca6c8006b344)

上图所示为标准卷积和Deformable卷积中的receptive field，采样位置(sampling locations)在整个顶部特征图(左)中是固定的。它们根据可变形卷积中对象的比例和形状进行自适应调整(右)。可以看到经过两层的传统卷积和两层的Deformable卷积的对比结果。左侧的传统卷积单个目标共覆盖了5 x 5=25个采样点，感受野始终是固定不变的方形；右侧的可变形卷积因为感受野的每一个点都有偏移量，造成卷积核在图片上滑动时对应的感受野的点不会重复选择，这意味着会采样9 x 9=81个采样点，比传统卷积更多。

> 显然，传统卷积核在卷积过程中由于会存在重叠，因此输出后的感受野范围小，**而可变性卷积中因为有偏移，不会有重叠，从而感受野范围更大**

**可变形卷积的优点：**

- 对物体的**形变和尺度建模的能力比较强。**
- **感受野比一般卷积大很多**，因为有偏移的原因，实际上相关实验已经表明了DNN网络很多时候受感受野不足的条件制约；但是一般的空洞卷积空洞是固定的，对不同的数据集不同情况可能最适合的空洞大小是不同的，但是可形变卷积的偏移是可以根据具体数据的情况进行学习的。

## Deformable DETR 原理分析
我们知道，DETR利用了Transformer通用以及强大的对相关性的建模能力，来取代anchor，proposal等一些手工设计的元素。但是DETR依旧存在2个缺陷：

**训练时间极长** ：相比于已有的检测器，DETR需要更久的训练才能达到收敛(500 epochs)，比Faster R-CNN慢了10-20倍。
**计算复杂度高** ：**发现DETR对小目标的性能很差**，现代许多种检测器通常利用多尺度特征，从高分辨率(High Resolution)的特征图中检测小物体。但是高分辨率的特征图会大大提高DETR复杂度。

**DETR网络是什么样的结构？**
产生上面两个问题的原因是：

- 在初始化阶段， **attention map 对于特征图中的所有pixel的权重是一致的，导致要学习的注意力权重集中在稀疏的有意义的位置这一过程需要很长时间**，意思是 attention map 从Uniform到Sparse and meaningful需要很久。
- attention map 是 Nq x Nk 的，在图像领域我们一般认为 **Nq = Nk = Nv = N = HW**, 所以里面的weights的计算是像素点数目的平方。 因此，处理高分辨率特征图需要非常高的计算量，存储也很复杂。

所以Deformable DETR的提出就是为了解决上面的两个问题，它主要利用了**可变形卷积** (Deformable Convolution)的**稀疏空间采样的本领**，**以及Transformer的对于全局关系建模的能力** 。针对此提出了一种 Deformable Attention Module ，这个东西**只关注一个feature map中的一小部分关键的位置**，起着一种pre-filter的作用 。这个 deformable attention module 可以自然地结合上**FPN ，我们就可以聚集多尺度特征。**

**这个模块可以聚合多尺度特征，不需要FPN了，我们用这个模块替换了Transformer Encoder中的Multi-Head Self- Attention模块和Transformer Decoder中的Cross Attention模块。**

Deformable DETR的提出可以帮助探索更多端到端目标检测的探索。提出了bbox迭代微调策略和两阶段方法，其中iterative bounding box refinement类似Cascade R-CNN方法，two stage类似RPN。

DETR探测小物体方面的性能相对较低，与现代目标检测相比，DETR需要更多的训练epoches才能收敛，这主要是因为处理图像特征的注意模块很难训练。所以本文提出了 Deformable Attention Module 。设计的初衷是：**传统的 attention module 的每个 Query 都会和所有的 Key 做attention，而 Deformable Attention Module 只使用固定的一小部分 Key 与 Query 去做attention，所以收敛时间会缩短。**

## Deformable Attention Module
首先对比下常规attention module与deformable在表达式上的不同：

![image](https://github.com/icey-zhang/notebook/assets/54712081/cc48b22b-2fec-405b-afdd-fd2f41275e3e)
![image](https://github.com/icey-zhang/notebook/assets/54712081/b0392592-2b61-411c-9f82-bcfe26432233)
![image](https://github.com/icey-zhang/notebook/assets/54712081/2b31f00c-7dde-4a5e-8283-d258e7949b30)

然后再比较一下二者的self-attention module的不同，如下图，可以发现：Deformable的Attention不是又Query和Key矩阵做内积得到的，而是由输入特征 i 直接通过Lienear Transformation得到的。

> Deformable 在具体的代码实现时，Encoder和Decoder的所有attention中除了decoder的self-attention利用Q,K,V三个矩阵，剩下的像encoder的self以及decoder的cross都是只使用了Q,V两个矩阵
![image](https://github.com/icey-zhang/notebook/assets/54712081/a3b0b0a2-f123-480f-a779-fe94b5c59150)

我们假设输入特征 i 的维度是 (Nq, C) ，他与几个转移矩阵相乘得到δx，δy以及A ，他们的维度都是 （Nq，M*K） ，其中A表示attention，后面会跟value为内积。

δx，δy代表相对参考点的偏移量，对于Encoder来将，Nq=H*W，即特征图上的每一个点都是向量，δx，δy表示的就是特征图上某点（x，y）对应的value的位置，因为value的维度是（HW, C）维的，所以有HW个位置可以对应，我们需要的就是其中K个位置。

同时，输入特征 i 再与转移矩阵W’ 相乘得到 Value∈（HW，C）矩阵，结合前面计算的δx，δy，可以为Nq维的query中的每一个向量都采样K个分量

- 为什么是query中的每一个维都取k个分量，这是因为我们从δx的维度就可以知道（Nq，K），而Attention的维度是（Nq, K）

所以采用之后的Value∈（Nq，K, C），M个head的Value就是（Nq，M，K, C），这样就可以使用Attention中的每一行与Value中的一组（M, K, C）去做weighted sum ，并把结果拼接到一起，得到的输出是O∈（Nq，M, C），最后将所有的head拼接到一起。

![image](https://github.com/icey-zhang/notebook/assets/54712081/ab7d76d2-4f47-45dc-843f-a3300d53dd87)
图2. Deformable Attention Module

Deformable Attention Module主要思想是结合了DCN和自注意力，目的就是为了通过在输入特征图上的参考点(reference point)附近只采样少数点(deformable detr设置为3个点)来作为注意力的$k$。因此要解决的问题就是：（1）确定reference point。（2）确定每个reference point的偏移量(offset)。（3）确定注意力权重矩阵 $A_{mqk}$。在Encoder和Decoder中实现方法不太一样，加下来详细叙述。

在**Encoder**部分，输入的Query Feature $z_q$为加入了位置编码的特征图(src+pos)， value(x)的计算方法只使用了src而没有位置编码(value_proj函数)。

（1）reference point确定方法为用了torch.meshgrid方法，调用的函数如下(get_reference_points)，有一个细节就是**参考点归一化到0和1之间**，因此取值的时候要用到**双线性插值**的方法。而在**Decoder**中，参考点的获取方法为object queries通过一个nn.Linear得到每个对应的reference point。

```python
def get_reference_points(spatial_shapes, valid_ratios, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # 从0.5到H-0.5采样H个点，W同理 这个操作的目的也就是为了特征图的对齐
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points
```

（2）计算offset的方法为对$z_q$过一个nn.Linear，得到多组偏移量，每组偏移量的维度为参考点的个数，组数为注意力头的数量。

（3）计算注意力权重矩阵$A_{mqk}$的方法为$z_q$过一个nn.Linear和一个F.softmax，得到每个头的注意力权重。

如图2所示，分头计算完的注意力最终会拼接到一起，然后最后过一个nn.Linear得到输入$x$的最终输出。

![image](https://github.com/icey-zhang/notebook/assets/54712081/2fdc11c8-160e-484a-bb53-457e843476d8)
![image](https://github.com/icey-zhang/notebook/assets/54712081/e019a5f1-c1c8-4302-81a2-028f9cf15174)

                        
原文链接：https://blog.csdn.net/qq_41439608/article/details/118249798
![image](https://github.com/icey-zhang/notebook/assets/54712081/55efd7f9-99cb-4071-93a7-dd21224794c3)


## Multi-scale Deformable Attention Module

![image](https://github.com/icey-zhang/notebook/assets/54712081/bd3a1d28-7955-4f89-937c-92bf4a926b8e)
式子中 L 表示feature map的level。

![image](https://github.com/icey-zhang/notebook/assets/54712081/8198dc64-6c1b-49cd-abe0-b651e9d17611)
- 这里不使用FPN，是因为每个query会与所有层的Key进行聚合

![image](https://github.com/icey-zhang/notebook/assets/54712081/ca02c859-1009-4f0c-81ef-1d933f756fc8)


![image](https://github.com/icey-zhang/notebook/assets/54712081/40264b76-ff45-4ad0-8d84-debdab7714a2)


![image](https://github.com/icey-zhang/notebook/assets/54712081/16e866d9-2ea0-4f3d-9d26-805fd6170e7a)

![image](https://github.com/icey-zhang/notebook/assets/54712081/488178f5-1337-433b-bc1d-c7b7bc397690)
![image](https://github.com/icey-zhang/notebook/assets/54712081/d3712bca-3778-48ee-b8f7-6957729c3974)


**参考**
[详解可变形注意力模块（Deformable Attention Module）](https://blog.csdn.net/qq_23981335/article/details/129123581)
[DETR 论文解读](https://blog.csdn.net/qq_38253797/article/details/127429466)
[Deformable DETR 论文+源码解读](https://blog.csdn.net/qq_38253797/article/details/127668593)
[Deformable DETR论文回稿](https://openreview.net/forum?id=gZ9hCDWe6ke)
[Deformable DETR论文精度+代码详解](https://zhuanlan.zhihu.com/p/596303361)
