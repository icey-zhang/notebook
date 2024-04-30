1.前言
RetinaNet是继SSD和YOLO V2公布后，YOLO V3诞生前的一款目标检测模型，出自何恺明大神的《Focal Loss for Dense Object Detection》。全文针对现有单阶段法（one-stage)目标检测模型中前景(positive)和背景(negatives)类别的不平衡问题，提出了一种叫做Focal Loss的损失函数，用来降低大量easy negatives在标准交叉熵中所占权重（提高hard negatives所占权重)。为了检测提出的Focal Loss损失函数的有效性，所以作者就顺便提出了一种简单的模型RetinaNet。（所以RetinaNet不是本篇论文的主角，仅仅是附属物了呗？）

本篇文章就来谈谈这个顺便被提出来的RetinaNet,代码是基于pytorch的，链接如下：
[code](https://link.zhihu.com/?target=https%3A//github.com/yhenon/pytorch-retinanet)

2. RetinaNet网络框架

![image](https://github.com/icey-zhang/notebook/assets/54712081/4e6a6bdb-0a21-4966-b642-180c03fdb308)

上图为RetinaNet的结构图，我们可以看出，RetinaNet的特征提取网络选择了残差网络ResNet，特征融合这块选择了FPN（特征金字塔网络），以特征金字塔不同的尺寸特征图作为输入，搭建三个用于分类和框回归的子网络。分类网络输出的特征图尺寸为（W,H,KA)，其中W、H为特征图宽高，KA为特征图通道，存放A个anchor各自的类别信息（K为类别数）。

2.1 残差网络ResNet

关于ResNet，这里进行一个简单又快速的介绍。ResNet的提出解决了由于网络过深而导致训练时出现梯度爆炸或者消失等问题。所以常见的ResNet一般从18层到152层（甚至更多）不等。他们的区别主要在于采用的残差单元/模块不同或者堆叠残差单元/模块的数量和比例不同。

![image](https://github.com/icey-zhang/notebook/assets/54712081/a235d2d9-1831-4401-907e-dde3af3b7b07)

RetinaNet代码中使用的是Resnet-50网络，该网络定义如下：
```python
model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
```
[3,4,6,3]为ResNet-50中不同通道残差单元的数量。

2.2 FPN网络

都2020年了，现在目标检测模型不来个特征融合都不好意思了，这里引用了EfficientDet中的一张图。可见，作为目标检测模型性能提升的一个点，FPN目前已经被研究的花里胡哨的。

![image](https://github.com/icey-zhang/notebook/assets/54712081/b194ac20-be32-4e4c-aaab-6605b44b4b3e)

当然了，RetinaNet刚提出的那会儿，FPN也没提出多久，所以中规中矩，RetinaNet用的图上的（a)那种自顶向下的FPN结构。采用FPN这种多尺度特征融合的目的，是为了对较小物体也能够保持检测的精度，就像SSD中的多尺度特征图一样（虽然他没有进行自顶向下的融合）。

代码中是这样实现的：

```python
class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest') #scale_factor为缩放大小
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
```
该特征金字塔网络输入为残差网络ResNet-50不同尺度的三张特征图；输出为多尺度特征融合的P3_x, P4_x, P5_x，以及由残差网络ResNet-50最后的一张特征图继续通过卷积层（stride=2,特征图尺寸继续减小）的结果P6_x, P7_x 。

2.3 框回归和分类子网络

RetinaNet在特征提取网络ResNet-50和特征融合网络FPN后，对获得的五张特征图[P3_x, P4_x, P5_x, P6_x, P7_x]，通过具有相同权重的框回归和分类子网络，获得所有框位置和类别信息。

框回归子网络定义如下：
```pyhon
class RegressionModel(nn.Module):
    """
    for bounding box regression 
    so the output size is 4.anchor.feature size"""
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        # 返回shape为（batch_size,anchor_nums,4）
        return out.contiguous().view(out.shape[0], -1, 4)
```
框回归子网络由四层卷积层组成，每层卷积层的stirde=1,kernel_size=3,padding=1，也就是说特征图通过该网络，长宽大小不变，通道维变为4*num_anchors。很明显，这里的通道维度的含义是：
> 保存了所有的框的位置信息（基于先验框/基础框anchor进行变换的位置信息）

同理，分类子网络定义如下：
```python
class ClassificationModel(nn.Module):
    """the same structure as bounding box regression subnet
    but the output channel is different from that in regression subnet """
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes * n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
```
和框回归子网络类似，分类子网络也是通过了四层卷积层，特征图的长宽保持不变，通道扩展为类别数 * anchor数，用于存放所有基于anchor的检测框的分类信息。

2.4 总模型定义

介绍完毕ResNet-50,FPN和框回归和分类子网络后，该网络的总模型定义如下，（代码中forward函数）：
```python
  def forward(self, inputs):

      if self.training:
          img_batch, annotations = inputs
      else:
          img_batch = inputs

      # Resnet
      x = self.conv1(img_batch)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x1 = self.layer1(x)
      x2 = self.layer2(x1)
      x3 = self.layer3(x2)
      x4 = self.layer4(x3)

      # FPN
      features = self.fpn([x2, x3, x4]) #pyramid feature

      # （batch_size,total_anchor_nums,4）
      # 框回归子网络
      regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

       # （batch_size,total_anchor_nums,class_num）
      #  框分类子网络
      classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
```
其中子网络的类实例化定义如下：
```python
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
```
3. 先验框anchor
每次解析基于anchor的目标检测模型，就一定要对它的anchor部分进行一个详细介绍，RetinaNet也不例外。前面提到的RetinaNet网络的输出为5张大小不同特征图，那么不同大小的特征图自然是负责不同大小物体检测（和特征图所对应的感受野相关）。

这里手动设置每个特征图对应的anchor基础框大小、缩放比例和长宽比，如下定义（类Anchors中)：
```python
if pyramid_levels is None:
    self.pyramid_levels = [3, 4, 5, 6, 7]
if strides is None:
    self.strides = [2 ** x for x in self.pyramid_levels] #[8, 16, 32, 64, 128]
if sizes is None: #base_size选择范围
    self.sizes = [2 ** (x + 2) for x in self.pyramid_levels] #[32,64,128,256,512]
if ratios is None:
    self.ratios = np.array([0.5, 1, 2])
if scales is None:
    self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
```

可以看出，对5个不同大小的特征图：
> [原图大小/8, 原图大小/16, 原图大小/32, 原图大小/64, 原图大小/128]

base_size设置为：
> [32,64,128,256,512]

即对于长宽为（原图大小/8，原图大小/8）的特征图，其特征图上的每个单元格cell对应原图区域上（32，32）大小的对应区域（如下图所示，这里对应的大小并不是实际感受野的大小，而是一种人为的近似设置）。

![image](https://github.com/icey-zhang/notebook/assets/54712081/990a176f-e934-4ba2-90f2-eca7fba95c60)

那么在大小为base_size的正方形框的基础上，对框进行长宽比例调整（3 种，分别为[0.5, 1, 2]）和缩放（3种，分别为[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]），便形成**9种所谓的基础框/先验框anchor**，如下图所示，（注意颜色对应），为一个单元格cell的通道维和anchor对应的关系。

![image](https://github.com/icey-zhang/notebook/assets/54712081/1c92ba76-ae0e-4318-bbab-4dd2ad51e330)

代码中实现所有anchor生成的程序为：

```python
def forward(self, image):
    
    image_shape = image.shape[2:]
    image_shape = np.array(image_shape)
    # 不同特征图上的size ,[原图/8，原图/16，原图/32，原图/64，原图/128]
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4)).astype(np.float32)

    for idx, p in enumerate(self.pyramid_levels):
        anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
        shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    all_anchors = np.expand_dims(all_anchors, axis=0)

    # anchor的数量随着图片分辨率增大而增大
    if torch.cuda.is_available():
        return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
    else:
        return torch.from_numpy(all_anchors.astype(np.float32))
```

，其中，对某一特定大小的特征图，生成其所有anchor坐标信息如下：

```python
def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        # based on the reference of the base_size ,like the operation of uniformization
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    # 复制成2行，3列 ,即（2，9）
    # 转置成（9，2），每行都是一组ratio和scale的组合，比例是base_size的
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors
```
其实上述函数的意思就是将上图的anchor按固定长度进行平移，然后和其对应特征图的cell进行对应，如下图所示。

![image](https://github.com/icey-zhang/notebook/assets/54712081/ff29efef-b411-4182-82be-736b4a1822e8)

这样，经过对每个特征图做类似的变换，生成全部anchor。

4. Focal Loss损失函数
我们知道，RetinaNet最主要的创新点在于其使用Focal Loss解决前景和背景样本不平衡的问题，那么这里就对Focal Loss进行详细介绍。代码中给出的FocalLoss类的forward函数，如下：

> def forward(self, classifications, regressions, anchors, annotations):

说明Focal Loss损失函数输入为网络的分类预测，框回归预测，anchor信息（用于对框回归预测进行转换）和真实标签annotations。

```python
alpha = 0.25
gamma = 2.0
batch_size = classifications.shape[0]
classification_losses = []
regression_losses = []

anchor = anchors[0, :, :]

anchor_widths  = anchor[:, 2] - anchor[:, 0]
anchor_heights = anchor[:, 3] - anchor[:, 1]
anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
```

该类前向函数先设定了一些基本参数，**然后将anchor由（左上坐标，右下坐标）转为（中心坐标，宽高）格式**。

接着对于batch_size中的每一张图片，做以下处理

```python
for j in range(batch_size):

    classification = classifications[j, :, :]
    regression = regressions[j, :, :]

    bbox_annotation = annotations[j, :, :]
    bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

    if bbox_annotation.shape[0] == 0:
        if torch.cuda.is_available():
            regression_losses.append(torch.tensor(0).float().cuda())
            classification_losses.append(torch.tensor(0).float().cuda())
        else:
            regression_losses.append(torch.tensor(0).float())
            classification_losses.append(torch.tensor(0).float())

        continue

    classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
```

接着**计算所有anchor与真实框的IOU大小**，并且找到所有anchor IOU最大的**真实框**的**索引**以及该**IOU大小**。
```python
IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
```

接着使用focal loss计算分类损失:

```python
# compute the loss for classification
targets = torch.ones(classification.shape) * -1 #(anchor_nums,class_num)

if torch.cuda.is_available():
    targets = targets.cuda()

targets[torch.lt(IoU_max, 0.4), :] = 0 #IOU<0.4为负样本，bool,（anchor_nums,1）

positive_indices = torch.ge(IoU_max, 0.5)##IOU>0.5为正样本，bool,（anchor_nums,1）

num_positive_anchors = positive_indices.sum()#正样本个数

assigned_annotations = bbox_annotation[IoU_argmax, :] #（anchor_nums,4）

targets[positive_indices, :] = 0
# assigned_annotations[positive_indices, 4] shape为（anchors_num,),每个元素为类别的索引序号
# 下面是转ONE-HOT编码的意思
targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

if torch.cuda.is_available():
    alpha_factor = torch.ones(targets.shape).cuda() * alpha
else:
    alpha_factor = torch.ones(targets.shape) * alpha

# 使用focal loss求解分类损失
alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor) #正负样本的权重不一样
focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

# cls_loss = focal_weight * torch.pow(bce, gamma)
cls_loss = focal_weight * bce

if torch.cuda.is_available():
    cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
else:
    cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
```

然后开始计算回归框的focal loss损失：
```python
# compute the loss for regression

if positive_indices.sum() > 0:
    assigned_annotations = assigned_annotations[positive_indices, :] #IOU>0.5的anchor,shape=（anchor_nums,4）

    anchor_widths_pi = anchor_widths[positive_indices]
    anchor_heights_pi = anchor_heights[positive_indices]
    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

    gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
    gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
    gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

    # clip widths to 1
    gt_widths  = torch.clamp(gt_widths, min=1)
    gt_heights = torch.clamp(gt_heights, min=1)

    # 将真实框映射到网络的输出空间
    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
    targets_dw = torch.log(gt_widths / anchor_widths_pi)
    targets_dh = torch.log(gt_heights / anchor_heights_pi)

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
    targets = targets.t()

    if torch.cuda.is_available():
        targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
    else:
        targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
```
上述步骤中主要通过anchor,将真实框的信息映射到RetinaNet网络的输出空间上（targets/labels)，同网络在前向过程中的输出（predictions）一起求解损失函数，这就和SSD中的match+encode是类似的。
最后根据上述映射到输出空间的target,使用focal loss求解损失函数：
```python
                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
```
至此，focal loss就讲完了，RetinaNet解析也接近尾声了。

5. 总结
6. 参考
[RetinaNet 论文和代码详解](https://zhuanlan.zhihu.com/p/143877125)
