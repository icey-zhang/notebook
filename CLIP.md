## 引言

相关工作 

- zero-shot迁移学习
  【Li 2017】用zero-shot迁移性的学习
  缺点：但是当时并没有transfomer的结构。在imagenet上做zero-shot只有11%的精度。
  transfomer出现后，【VirTex 2020】自回归的方式，【ICMLM 2020】完形填空方式，【ConVIRT 2020】只在医学图像上进行验证
  优点：图片和文字结合起来去学得更好的特征
  缺点：但是没有数据集和模型的扩大

- 弱监督的信号，数据集上的改进
  【Mahajan 2018】instagram的数据集，每张图片都会带hashtag，相当于是图片的标注，自然语言的监督信号，有明确的语义含义。
  【Kolesnikov 2019】【Dosovitskiy 2020】JFT-300M数据集比较大，类别比较多1.8w，所以标签比较noisy。相当于是弱监督信号。
  优点：文本带来的弱监督的信号去训练有监督的模型。精度比较高
  缺点：用softmax的分类头去做分类，所以只能输出固定类别，缺乏灵活性，限制了zero-shot的能力

【Mahajan 2018】和【Kolesnikov 2019】使用比较大规模的数据集，但是【VirTex 2020】【ICMLM 2020】【ConVIRT 2020】只使用了比较小的数据集

【ConVIRT 2020】的简化版本
尝试了CV的8个模型，模型大小差几百倍，模型增大，迁移学习效果越好，比较平滑的相关性。刷了30个数据集。

## 方法

### 2.1 自然语言监督的优势

【Zhang 2020】、【 Gomez 2017】、【Joulin 2016】、【Desai&Johnson】定义比较混淆，无监督，自监督，监督。

自然语言的监督信号来训练一个视觉模型。文本当作一个训练的信号 

NLP领域 topic modl 和 n-gram 的训练方式较为复杂具有上下文语义信息的表征学习方式代表着我们现在可以有工具，更有效的利用文本监督的大量资源。

**为什么可以用自然语言监督信号训练视觉模型？**

1.不需要标注，只需下载图片和文字的配对

比如ImageNet需要先定义好1000个类，然后根据这些类去下载图片，清理数据集，再去标注所有图片，过程很复杂。而`CLIP`不要求这种经典的“”机器学习兼容“”的标注格式，只需要下载文字-图片对；且没有n选1的标签之后，模型的输入输出自由度大了很多。

2.学到的特征不只是一个视觉特征，而是一个多模态特征，

MAE，moco的自监督学习只能学到视觉特征

### 2.2 创建一个足够大的数据集

清洗之后的数据集规模小了6倍数，只有15milliom图片。

4个亿的数据集，和训练GPT2的webtext大小差不多

### 2.3 有效的预训练方法

【Mahajan 2018】需要19个GPU年

【Xie 2020】需要噢33个TPU年

类似于VirTex 图片用CNN，文本用transfomer，预测性的任务会导致预测结果有很多的可能性，导致训练困难。

对比学习的方式可以提高效率

CV领域的模型都很大，训练起来也很贵。比如noise student之前在ImageNet一直霸榜，但是这个模型需要在一个 TPUv3上训练33年，这还只是在包含1000类的ImageNet上预训练的，而且只训练视觉特征。
  由于训练数据量和模型计算量都很大，训练效率成为一个至关重要的因素。作者做了很多尝试，最终选择了对比学习：

- VirTex模型：预测文本，对应下图蓝色线Transformer Language Model
  Image Encoder使用CNN模型，Text Encoder使用transformer模型，两个模型一起从头训练，任务是预测图片对应的文本（image caption）。
  这种方法的训练效率太慢，因为根据图片进行文本描述，可能性太多了，你可以从各个角度去描述一张图片。
- Bag of Words Prediction（橘色线）：不要求每个词都是按顺序的进行预测，所有词都预测出来就行。这样放宽了约束，训练速度提高了三倍。
- CLIP：简化版的ConVIRT，基于对比学习。
  只需要判断图文是否配对，进一步简化了训练任务，训练效率一下子提升4倍（绿色线）
  训练任务更加合理。因为训练数据所包含的文本-图像对是从互联网收集来的，它们存在一定的噪音，二者并不完全匹配。适当的降低训练目标，反而能取得更好的收敛。

![image-20240719123702565](/Users/zhangjiaqing/Library/Application Support/typora-user-images/image-20240719123702565.png)

损失函数的创建

![image-20240719124121396](/Users/zhangjiaqing/Library/Application Support/typora-user-images/image-20240719124121396.png)

在MOCO中，真实标签都是0，因为其正样本都是放在第一位，所以正样本对应的索引永远是0；但是在CLIP中，正样本都是在对角线上，所以真实标签为np.arange(n)。
对称式的目标函数，simiCLR mocov3 DINO相似

**训练细节**

- 收集的数据集大不会存在overfiting的情况。

- 文本编码器和视觉编码器不需要预训练

- 非线性的投射层适配纯图片的单模态学习（simiCLR mocov3），只使用线性投射层（线性非线性影响不大）

- 数据集比较大，只用了随机裁剪的方式

- 不好做调参工作，temperature是一个可学习的标量

### 2.5 训练

- 对于视觉训练了8个模型，5个resnet和3个vision transfomer
     - Resnet50，Resnet101和类似于Efficient方式的修改模型的深度和输入大小（Resnet50x4，Resnet50x16，Resnet50x64）数字表示计算量
     - ViT-B/32、ViT-B/16、ViT-L/14 数字表示patch的大小

- 32epoch/Adam，batch-size=32768

- 用resnet50一个epoch调参

[How to Train Really Large Models on Many GPUs? | Lil'Log (lilianweng.github.io)](https://lilianweng.github.io/posts/2021-09-25-train-large/)

- 分布式训练，混精度训练

- RN50x64在592 V100 GPU上训练了18天，ViT-L在256 V100上训练了12天。ViT-L/14在更大336像素尺寸图片上fine-tine了一个epoch（ViT-L/14@336）

## 实验

### 3.1 zero-shot 迁移

研究zero-shot的动机：之前的自监督或有监督训练的模型（MOCO、DINO等），主要是学习一种泛化好的特征，所以在做下游任务的时候，还是需要有监督的微调，就依然存在很多问题。比如下游任务的数据集不好收集，存在分布飘偏移（distribution shift）等等。而使用文本引导视觉模型训练，就可以很好的进行zero-shot迁移；模型就可以不再训练，不再微调。#

### 3.2 Prompt Engineering and Ensembling

#### **Prompt Engineering**

作者还验证了文本描述时采用prompt的有效性（精度提升1.3%）。简单来说，prompt learning的核心是通过构建合适prompt（提示）来使预训练模型能够直接应用到下游任务中。

推理时，只使用类别标签作为文本描述效果并不够好，原因有二：

1.词语存在歧义性
如果我们直接采用类别标签作为文本描述，那么很多文本就是一个单词，缺少具体的上下文，并不能很好的描述图片内容。

- 比如在做物体检测时，有一个类别是remote（遥控器）。但如果直接喂给文本编码器，很可能被模型认为是遥远的意思。

- 同一个词语在不同数据集中所表示的意思可能有所不同。例如在 Oxford-IIIT Pets 数据集中，boxer指的是狗的一个种类，在其他数据集中指的是拳击运动员。

- 所以 CLIP预训练时，用来描述图片内容的文本是一个句子，比如A photo of {label}。这里的label就只能是名词，一定程度上消除了歧义性。

2.使推理和预训练时保持一致（消除distribution gap）。

  另外，还可以根据不同的数据集来调整这个模板，进而提升zero-shot的性能。
  例如当数据集是Oxford-IIIT Pets数据集时（类别都是动物），就可以将模板写成： A photo of a {label}, a type of pet. ；或者在做OCR任务时，在想找的那个文本或者数字上打上双引号，模型就可能知道你是想找双引号里面的内容。

#### prompt ensembling

​        作者尝试了集成多个模板的效果，即在多个zero-shot分类器上进行集成，这些分类器使用不同的提示模板来构造不同的文本。由于是在嵌入空间(embedding space)而不是概率空间(probability space)上集成的，因此节约了计算成本。在大多数数据集上，prompt ensembling都能够提升模型性能。

  最终作者使用了80种模板来进行集成，每种模板使用了不同的形容词，来，描述不同的情境。
![在这里插入图片描述](https://img-blog.csdnimg.cn/02191521495041acb0528faf4524466f.png)

上图横坐标表示模型算力，纵坐标表示在多个数据集上的平均分数。绿色曲线表示本文中使用Prompt engineering and ensembling的结果，蓝色曲线表示直接使用无提示上下文的类名的结果。

### 3.3 zero-shot分类效果对比（ResNet-50）

  为了测试CLIP的zero-shot分类的效果怎么样，作者将在27个数据集上的分类效果做成了对比图，下图就是CLIP与基于ResNet-50做Linear Probe的对比。

![在这里插入图片描述](https://img-blog.csdnimg.cn/dc5e9391eca4447e9a5d6d1778ef1115.png)

- Linear Probe on ResNet-50：
  - Linear Probe就是冻住预训练好的模型，只训练最后一层的分类器，相当于将预训练模型做特征提取器。
  - ResNet50是在ImageNet上用有监督的方式预训练好的

- 对比结果：
  - 绿色 + 表示相比ResNet-50提升了多少，蓝色 - 表示相比ResNet-50降低了多少。
  - 最终在27个数据集中，CLIP在16个数据集上都超越了有监督训练好的ResNet-50。
  - 对于普通的物体分类任务，CLIP可以很好的做zero-shot迁移，例如车、食物、CIFAR10等数据集，因为图像中有可以描述出来的物体，那对应的文本中也就有这种描述，因此可以很好的匹配；
  - 但CLIP对于更加复杂或抽象的任务就表现比较弱，例如卫星图像分类、淋巴结肿瘤检测等需要特定领域知识的分类任务，CLIP并没有预训练到这些标签信息。

### 3.4 few-shot分类效果对比

  作者认为，这种特别难的任务，完全不给任何标签信息，有点强人所难了，不是很合理。所以论文还对比few-shot性能，即只用少量的样本来微调模型，这里对比了3个模型：

- 在ImageNet21K上训练的BiT-M （big transfer），是一个很强的baseline。
- 基于SimCLRv2训练的ResNet50，
- 有监督训练的ResNet50。
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/915bf51100b94613b5013eb4089363f1.png)

- 横坐标：每个数据集每个类别里，用了多少个标注样本进行Linear Probe的分类器训练。0就相当于zero-shot了。
- 纵坐标表示在20个数据集上的平均分类准确度（有7个数据集每个类别不够16个）
- 当每类有16个训练样本时，BiT-M模型的性能才和zero-shot CLIP打成平手。
- 紫色曲线说明：每类的训练样本只有1个或2个的时候，效果还不如zero-shot CLIP；但当每类的训练样本增加到8个或16个的时候，效果则超越了zero-shot CLIP。这说明对于一些难的数据集来说，有一些训练样本还是非常有必要的。
  
  

参考

[李沐论文精读系列四：CLIP和改进工作串讲（LSeg、GroupViT、VLiD、 GLIPv1、 GLIPv2、CLIPasso）_n-vocabulary object detection via vision and langu-CSDN博客](https://blog.csdn.net/qq_56591814/article/details/127421979)

[CLIP 论文逐段精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1SL4y1s7LQ/?spm_id_from=333.999.0.0&vd_source=ee28f748a7042b99cf81403720f8106e)
