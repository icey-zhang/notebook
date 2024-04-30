本文目录
>1 BERT 方法回顾

>2 BERT 可以直接用在视觉任务上吗？

>3 BEiT 原理分析
- 3.1 将图片表示为 image patches
- 3.2 将图片表示为 visual tokens
- 3.2.1 变分自编码器 VAE
- 3.2.2 BEIT 里的 VAE：tokenizer 和 decoder
- 3.2.3 BEIT 的 Backbone：Image Transformer
- 3.2.4 类似 BERT 的自监督训练方式：Masked Image Modeling
- 3.2.5 BEIT 的目标函数：VAE 视角
- 3.2.6 BEIT 的架构细节和训练细节超参数
- 3.2.7 BEIT 在下游任务 Fine-tuning
- 3.2.8 实验


Self-Supervised Learning，又称为自监督学习，我们知道一般机器学习分为有监督学习，无监督学习和强化学习。 而 Self-Supervised Learning 是无监督学习里面的一种，主要是希望能够学习到一种通用的特征表达用于下游任务 (Downstream Tasks)。 其主要的方式就是通过自己监督自己。作为代表作的 kaiming 的 MoCo 引发一波热议， Yann Lecun也在 AAAI 上讲 Self-Supervised Learning 是未来的大势所趋。所以在这个系列中，我会系统地解读 Self-Supervised Learning 的经典工作。


今天介绍的这篇工作 BEiT 是把 BERT 模型成功用在 image 领域的首创，也是一种自监督训练的形式，所以取名为视觉Transformer的BERT预训练模型。这个工作用一种巧妙的办法把 BERT 的训练思想成功用在了 image 任务中，涉及的知识点包括 BERT (第1节)，VAE (第3.2.1节) 等等，为了方便阅读本文也会对它们进行简单讲解。

总结下 Self-Supervised Learning 的方法，用 4 个英文单词概括一下就是：

**Unsupervised Pre-train, Supervised Fine-tune.**
下面首先借助 BERT 模型理解一下这句话的意思。

[什么是BERT？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/98855346)

![image](https://github.com/icey-zhang/notebook/assets/54712081/8d62b0b2-6188-408c-8572-3be37227868d)


BERT的全称为**Bidirectional Encoder Representation from Transformers**，是一个预训练的语言表征模型。它强调了不再像以往一样采用传统的单向语言模型或者把两个单向语言模型进行浅层拼接的方法进行预训练，而是采用新的**masked language model（MLM）**，以致能生成深度的双向语言表征。BERT论文发表时提及在11个NLP（Natural Language Processing，自然语言处理）任务中获得了新的state-of-the-art的结果。该模型有以下主要优点：
（1）采用MLM对双向的Transformers进行预训练，以生成深层的双向语言表征。
（2）预训练后，只需要添加一个额外的输出层进行fine-tune，就可以在各种各样的下游任务中取得state-of-the-art的表现。在这过程中并不需要对BERT进行任务特定的结构修改。

1.BERT的结构
以往的预训练模型的结构会受到单向语言模型（从左到右或者从右到左）的限制，因而也限制了模型的表征能力，使其只能获取单方向的上下文信息。而BERT利用MLM进行预训练并且采用深层的双向Transformer组件（单向的Transformer一般被称为Transformer decoder，其每一个token（符号）只会attend到目前往左的token。而双向的Transformer则被称为Transformer encoder，其每一个token会attend到所有的token。）来构建整个模型，因此最终生成能融合左右上下文信息的深层双向语言表征。
当隐藏了Transformer的详细结构后，我们就可以用一个只有输入和输出的黑盒子来表示它了：

![image](https://github.com/icey-zhang/notebook/assets/54712081/0d97e29a-ece0-4753-bef1-d1e86908d31e)

而Transformer结构又可以进行堆叠，形成一个更深的神经网络（这里也可以理解为将Transformer encoder进行堆叠）：

![image](https://github.com/icey-zhang/notebook/assets/54712081/a846f4ed-4393-43cf-b592-43b4df2a4d05)


最终，经过多层Transformer结构的堆叠后，形成BERT的主体结构：（中间蓝色的部分就是由多个Transformer结构所堆叠在一起的）

![image](https://github.com/icey-zhang/notebook/assets/54712081/7c75fe86-95a9-4681-86a7-7ded610cd7c8)


对于不同的下游任务，BERT的结构可能会有不同的轻微变化，因此接下来只介绍预训练阶段的模型结构。

（1）BERT的输入
BERT的输入为每一个token对应的表征（图中的粉红色块就是token，黄色块就是token对应的表征），并且单词字典是**采用WordPiece算法**来进行构建的。为了完成具体的分类任务，除了单词的token之外，作者还在输入的每一个序列开头都插入特定的**分类token（[CLS]）**，**该分类token对应的最后一个Transformer层输出被用来起到聚集整个序列表征信息的作用。**
由于BERT是一个预训练模型，其必须要适应各种各样的自然语言任务，因此模型所输入的序列必须有能力包含一句话（文本情感分类，序列标注任务）或者两句话以上（文本摘要，自然语言推断，问答任务）。那么如何令模型有能力去分辨哪个范围是属于句子A，哪个范围是属于句子B呢？BERT采用了两种方法去解决：
**①在序列tokens中把分割token（[SEP]）插入到每个句子后，以分开不同的句子tokens。
②为每一个token表征都添加一个可学习的分割embedding来指示其属于句子A还是句子B。**
因此最后模型的输入序列tokens为下图（如果输入序列只包含一个句子的话，则没有[SEP]及之后的token）：

![image](https://github.com/icey-zhang/notebook/assets/54712081/999c2a66-4e81-4216-9760-520c1797ca9e)



	上面提到了BERT的输入为每一个token对应的表征，实际上该表征是由三部分组成的，分别是对应的token，分割和位置 embeddings。如下图：

![image](https://github.com/icey-zhang/notebook/assets/54712081/bfd3f6a4-4a99-4b3b-bf4a-f27eeea8b1b5)


（2）BERT的输出
介绍完BERT的输入，实际上BERT的输出也就呼之欲出了，因为Transformer的特点就是有多少个输入就有多少个对应的输出：

![image](https://github.com/icey-zhang/notebook/assets/54712081/3ae32a39-c3c2-421b-bccf-c30c36fa43d1)


C为分类token（[CLS]）对应最后一个Transformer的输出， $T_i$则代表其他token对应最后一个Transformer的输出。对于一些token级别的任务（如，序列标注和问答任务），就把$T_i$输入到额外的输出层中进行预测。对于一些句子级别的任务（如，自然语言推断和情感分类任务），就把C输入到额外的输出层中，这里也就解释了为什么要在每一个token序列前都要插入特定的分类token。

2.BERT的预训练任务
虽然NLP领域没有像ImageNet这样质量高的人工标注数据，但是可以利用大规模文本数据的自监督性质来构建预训练任务。因此BERT构建了**两个预训练任务**，**分别是Masked Language Model和Next Sentence Prediction。**

（1）Masked Language Model （MLM）
	MLM是BERT能够不受单向语言模型所限制的原因。简单来说就是以15%的概率用mask token （[MASK]）随机地对每一个训练序列中的token进行替换，然后预测出[MASK]位置原有的单词。然而，由于[MASK]并不会出现在下游任务的微调（fine-tuning）阶段，因此预训练阶段和微调阶段之间产生了不匹配（这里很好解释，就是预训练的目标会令产生的语言表征对[MASK]敏感，但是却对其他token不敏感）。因此BERT采用了以下策略来解决这个问题：

首先在每一个训练序列中以15%的概率随机地选中某个token位置用于预测，假如是第i个token被选中，则会被替换成以下三个token之一：
①80%的时候是[MASK]。如，my dog is hairy——>my dog is [MASK]
②10%的时候是随机的其他token。如，my dog is hairy——>my dog is apple
③10%的时候是原来的token（保持不变）。如，my dog is hairy——>my dog is hairy

再用该位置对应的  去预测出原来的token（输入到全连接，然后用softmax输出每个token的概率，最后用交叉熵计算loss）。
该策略令到BERT不再只对[MASK]敏感，而是对所有的token都敏感，以致能抽取出任何token的表征信息。

（2）Next Sentence Prediction（NSP）
一些如问答、自然语言推断等任务需要理解两个句子之间的关系，而MLM任务倾向于抽取token层次的表征，因此不能直接获取句子层次的表征。为了使模型能够有能力理解句子间的关系，BERT使用了NSP任务来预训练，简单来说就是预测两个句子是否连在一起。具体的做法是：对于每一个训练样例，我们在语料库中挑选出句子A和句子B来组成，50%的时候句子B就是句子A的下一句（标注为IsNext），剩下50%的时候句子B是语料库中的随机句子（标注为NotNext）。接下来把训练样例输入到BERT模型中，用[CLS]对应的C信息去进行二分类的预测。

（3）预训练任务总结
最后训练样例长这样：
Input1=[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]
Label1=IsNext

Input2=[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label2=NotNext

把每一个训练样例输入到BERT中可以相应获得两个任务对应的loss，再把这两个loss加在一起就是整体的预训练loss。（也就是两个任务同时进行训练）
可以明显地看出，这两个任务所需的数据其实都可以从无标签的文本数据中构建（自监督性质），比CV中需要人工标注的ImageNet数据集可简单多了。

![image](https://github.com/icey-zhang/notebook/assets/54712081/495bfd4f-0426-4664-a863-8ba5ea3a9d63)


1 BERT 方法回顾

在 [Self-Supervised Learning 超详细解读 (一)：大规模预训练模型BERT](https://zhuanlan.zhihu.com/p/378360224) 里面我们介绍了 BERT 的自监督预训练的方法，BERT 可以做的事情也就是Transformer 的 Encoder 可以做的事情，就是输入一排向量，输出另外一排向量，输入和输出的维度是一致的。那么不仅仅是一句话可以看做是一个sequence，一段语音也可以看做是一个sequence，甚至一个image也可以看做是一个sequence。所以BERT其实不仅可以用在NLP上，还可以用在CV里面。所以BERT其实输入的是一段文字，如下图1所示。

![image](https://github.com/icey-zhang/notebook/assets/54712081/5d33061a-1f4e-4bcb-b9a9-52fbb29f146a)
图1：BERT的架构就是Transformer 的 Encoder


接下来要做的事情是把这段输入文字里面的一部分随机盖住。随机盖住有 2 种，一种是直接用一个Mask 把要盖住的token (对中文来说就是一个字)给Mask掉，具体是换成一个特殊的字符。另一种做法是把这个token替换成一个随机的token。

![image](https://github.com/icey-zhang/notebook/assets/54712081/7778e080-82dd-419e-8371-43cfb5f49b70)
图2：把这段输入文字里面的一部分随机盖住

接下来把这个盖住的token对应位置输出的向量做一个Linear Transformation，再做softmax输出一个分布，这个分布是每一个字的概率，如下图3所示。

那接下来要怎么训练BERT呢？因为这时候BERT并不知道被 Mask 住的字是 "湾" ，但是我们知道啊，所以损失就是让这个输出和被盖住的 "湾" 越接近越好，如下图4所示。

![image](https://github.com/icey-zhang/notebook/assets/54712081/389b5d30-dcc2-4b2a-b162-d4eea5880fd0)
图3：把这个盖住的token对应位置输出的向量做一个Linear Transformation

![image](https://github.com/icey-zhang/notebook/assets/54712081/d2662e61-0573-49d2-91be-09a04eed161c)
图4：让这个输出和被Mask 住的 token 越接近越好

其实BERT在训练的时候可以不止是选取一个token，我们可以选取一排的token都盖住，这就是 SpanBERT 的做法，至于要盖住多长的token呢？SpanBERT定了一个概率的分布，如图5所示。有0.22的概率只盖住一个token等等。
![image](https://github.com/icey-zhang/notebook/assets/54712081/e4506140-2401-47da-b497-f930e1305b21)
图5：SpanBERT定了一个概率的分布

除此之外，SpanBERT还提出了一种叫做Span Boundary Objective (SBO) 的训练方法，如下图6所示，意思是说：
![image](https://github.com/icey-zhang/notebook/assets/54712081/8228e7b8-4c02-409d-a5b8-6e38da417f28)
图6：Span Boundary Objective (SBO)
盖住一串token以后，用这段被盖住的token的左右2个Embedding去预测被盖住的token是什么。SBO把盖住的部分的左右两边的Embedding吃进来，同时还输入一个数字，比如说3，就代表我们要还原被盖住的这些token里面的第3个token。

就是通过上面的图1-图6的方法，让 BERT 看很多的句子，随机盖住一些 tokens，让模型预测盖住的tokens是什么，不断计算预测的 token 与真实的 token 之间的差异，利用它作为 loss 进行反向传播更新参数，来达到 Self-Supervised Learning 的效果。

Self-Supervised Learning 训练好 BERT 以后，如何在下游任务上使用呢？

我们就以情感分析为例，要求输入一个句子，输出对应的情感类别。

BERT是怎么解Sentiment Analysis的问题呢？给它一个句子，在这个句子前面放上 class token，这步和 ViT 是一模一样的。同样地，我们只取输出的Sequence里面的class token对应的那个vector，并将它做Linear Transformation+Softmax，得到类别class，就代表这个句子的预测的情感，如下图7所示。

值得注意的是，对于这种下游任务你需要有labelled data，也就是说 BERT 其实没办法凭空解Sentiment Analysis的问题，也是需要一部分有监督数据的。我们此时的情感分析模型包括：

BERT部分
Linear Transformation部分
只是BERT部分的初始化来自 Self-Supervised Learning，而 Linear Transformation 部分采样的是随机初始化。这两部分的参数都用Gradient Descent来更新。

![image](https://github.com/icey-zhang/notebook/assets/54712081/44a53924-898e-45b9-829b-93c76e2d83f3)
图7：使用BERT做情感分析

下图8其实是个对比，就是BERT部分不用预训练模型的初始化 (scratch) 和用了预训练模型的初始化 (fine-tune) 的不同结果，不同颜色的线代表GLUE中的不同任务。 不用预训练模型的初始化会导致收敛很慢而且loss较高，说明预训练模型的初始化的作用。

![image](https://github.com/icey-zhang/notebook/assets/54712081/9410dad8-e9af-420d-aa6a-d1e748b68c95)
图8：预训练模型的初始化结果

2 BERT 可以直接用在视觉任务上吗？
上面的 BERT 都是在 NLP 任务上使用，因为 NLP 任务可以把每个词汇通过 Word2Vec 自动转化成一个固定大小的 token，我们随机盖住一些 token，让模型根据这个不完整的句子来预测被盖住的 token 是什么。那么一个自然而然的问题是：对于图片来讲，能否使用类似的操作呢？

第1个困难的地方是：视觉任务没有一个大的词汇表。在 NLP 任务中，比如图3所示，假设我们盖住词汇 "湾"，那么就想让模型根据这个不完整的句子来预测被盖住的 token 是 "湾"，此时我们有个词汇表，比如这个词汇表一共有8个词，"湾" 是第3个，则 "湾" 这个 token 的真值就是GT=[0,0,1,0,0,0,0,0] 
 ，只需要让模型的输出和这个GT 越接近越好。

但是 CV 任务没有这个词汇表啊，假设我盖住一个 patch，让模型根据这个不完整的 image 来预测被盖住的 patch 是什么。那么对应的这个GT是什么呢？

BEIT 通过一种巧妙的方式解决了这个问题。

假设这个问题可以得到解决，我们就能够用 masked image modeling 的办法 (和BERT类似，盖住图片的一部分之后预测这部分) 训练一个针对图片的预训练模型，这个预训练模型就也可以像 BERT 一样用在其他各种 CV 的下游任务中啦。

3 BEIT 原理分析
> 论文名称：BEIT: BERT Pre-Training of Image Transformers

本文提出的这个方法叫做 BEIT，很明显作者是想在 CV 领域做到和 NLP 领域的 BERT 一样的功能。在第1篇文章中提到，训练好的 BERT 模型相当于是一个 Transformer 的 Encoder，它能够把一个输入的 sentence 进行编码，得到一堆 tokens。比如输入 "台湾大学"，通过 BERT 以后会得到4个 tokens。并且这4个 tokens 也结合了sentence 的上下文。

那 BEIT 能不能做到类似的事情呢？，即能够把一个输入的 image 进行编码，得到一堆 vectors，并且这些个 vectors 也结合了 image 的上下文。

答案是肯定的。BEIT 的做法如下：

在 BEIT 眼里，图片有 2 种表示的形式：

> image → image patches | visual tokens
在预训练的过程中，它们分别被作为模型的输入和输出，如下图9所示。
![image](https://github.com/icey-zhang/notebook/assets/54712081/973f8b55-2634-4132-a6f6-e51f46a23df7)
图9：图片有 2 种表示的形式：image patches or visual tokens

BEIT的结构可以看做2部分，分别是：

- BEIT Encoder
- dVAE

BEIT Encoder 类似于 Transformer Encoder，是对输入的 image patches 进行编码的过程，dVAE 类似于 VAE，也是对输入的 image patches 进行编码的过程，它们的=具体会在下面分别详细介绍。

3.1 将图片表示为 image patches
![image](https://github.com/icey-zhang/notebook/assets/54712081/a61ca686-0f10-4696-a993-5ca579f9c9fb)


问：image patch 是个扮演什么角色？

答：image patch 只是原始图片通过 Linear Transformation 的结果，所以只能保留图片的原始信息 (Preserve raw pixels)。

3.2 将图片表示为 visual tokens
![image](https://github.com/icey-zhang/notebook/assets/54712081/72829064-3477-46fb-9c1e-8f748feea3dd)

要彻底理解如何将图片表示为 visual tokens，那就得先从 VAE 开始讲起了，熟悉 VAE 的同学可以直接跳过3.2.1。

3.2.1 变分自编码器 VAE
VAE 跟 GAN 的目标基本是一致的——希望构建一个从隐变量$Z$生成目标数据$X$的模型，但是实现上有所不同。更准确地讲，它们是假设了$Z$服从某些常见的分布（比如正态分布或均匀分布），然后希望训练一个模型$X=G(Z)$，如下图10所示，这个模型能够将原来的概率分布映射到训练集的概率分布，也就是说，它们的目的都是进行分布之间的变换。

![image](https://github.com/icey-zhang/notebook/assets/54712081/069ef0bd-d119-4ba1-be39-31cfd6eeebd2)
图10：生成模型的难题就是判断生成分布与真实分布的相似度，因为我们只知道两者的采样结果，不知道它们的分布表达式

![image](https://github.com/icey-zhang/notebook/assets/54712081/5088ba3a-4191-4f2c-8bd7-c6b671a703dc)
图11：VAE的传统理解

![image](https://github.com/icey-zhang/notebook/assets/54712081/123e5f7e-e804-4fd1-b746-c139759928c4)

![image](https://github.com/icey-zhang/notebook/assets/54712081/1d785290-a923-4c6c-9156-de0892f5d94a)
图12：事实上，vae是为每个样本构造专属的正态分布，然后采样来重构

![image](https://github.com/icey-zhang/notebook/assets/54712081/16334db5-de5e-4361-9fd4-2144b6b1ffb6)

![image](https://github.com/icey-zhang/notebook/assets/54712081/e624022e-646c-41f6-81c0-35d86efebe40)
图13：均值方差通过一个神经网络来拟合出来

![image](https://github.com/icey-zhang/notebook/assets/54712081/0eb39205-5827-43d9-8462-64f015dae6c6)
![image](https://github.com/icey-zhang/notebook/assets/54712081/9f9f9913-d9d6-49d3-900c-ae54ce621c43)
![image](https://github.com/icey-zhang/notebook/assets/54712081/fcc4c069-1af1-4c3b-b7a3-015e94806f82)

3.2.2 BEIT 里的 VAE：tokenizer 和 decoder
上面我们了解了 VAE 模型的训练过程，那么我们回到之前的问题上面，BEIT 是如何将图片表示为 visual tokens的呢？

具体而言，作者训练了一个 discrete variational autoencoder (dVAE)。训练的过程如下图14所示。读者可以仔细比较一下这个 dVAE 和上文介绍的 VAE 的异同，dVAE 虽然是离散的 VAE，但它和 VAE 的本质还是一样的，都是把一张图片通过一些操作得到隐变量，再把隐变量通过一个生成器重建原图。下表就是一个清晰的对比，我们可以发现：

VAE使用图13所示的均值方差拟合神经网络得到隐变量。
dVAE使用Tokenizer得到隐变量。
VAE使用图12所示的生成器重建原图。
dVAE使用Decoder重建原图。

![image](https://github.com/icey-zhang/notebook/assets/54712081/4632e06e-1e62-46ed-8b9c-4b26df800ca9)
![image](https://github.com/icey-zhang/notebook/assets/54712081/f5cbe750-d634-47fc-97fd-870266c9ceb2)
图14：训练 discrete variational autoencoder (dVAE) 的过程

所以dVAE中的Tokenizer就相当于是VAE里面的均值方差拟合神经网络，dVAE中的Decoder就相当于是VAE里面的生成器。

所以，dVAE 的训练方式其实可以和 VAE 的一模一样。

问：这里的 visual token 具体是什么形式的？

答：作者把一张 224×224 的输入图片通过 Tokenizer 变成了 14×14 个 visual token，每个 visual token 是一个位于[1,8192]之间的数。就像有个 image 的词汇表一样，这个词汇表里面有 8192 个词，每个 16×16 的image patch会经过 Tokenizer 映射成 $|V|$里面的一个词。**因为 visual token 是离散的数，所以优化时没法求导，所以作者采用了 gumbel softmax 技巧**，想详细了解 gumbel softmax trick 的同学可以参考下面的链接：

 [gumbel softmax 技巧](https://zhuanlan.zhihu.com/p/166632315)

 3.2.3 BEIT 的 Backbone：Image Transformer
![image](https://github.com/icey-zhang/notebook/assets/54712081/d079f518-d7c0-4f84-91fd-18405967f445)

 ![image](https://github.com/icey-zhang/notebook/assets/54712081/0b4d1dbf-ec5a-47bd-8d45-2fc31730900e)
图15：BEIT 的总体结构
![image](https://github.com/icey-zhang/notebook/assets/54712081/fcd8a0b1-0c5b-435a-9b22-c22e3f0666e3)

3.2.4 类似 BERT 的自监督训练方式：Masked Image Modeling
至此，我们介绍了 BEIT 的两部分结构：

- **BEIT Encoder**
- **dVAE**
下面就是 BEIT 的训练方法了。既然BEIT 是图像界的 BERT 模型，所以也遵循着和 BERT 相似的自监督训练方法。BERT 的自监督训练方法忘了的同学请再看一遍图1-图6。

> 让 BERT 看很多的句子，随机盖住一些 tokens，让 BERT 模型预测盖住的tokens是什么，不断计算预测的 token 与真实的 token 之间的差异，利用它作为 loss 进行反向传播更新参数，来达到 Self-Supervised Learning 的效果。
BEIT 使用了类似 BERT 的自监督训练方式：Masked Image Modeling，如图15所示，即：

> 让 BEIT 看很多的图片，随机盖住一些 image patches，让 BEIT 模型预测盖住的patches是什么，不断计算预测的 patches 与真实的 patches 之间的差异，利用它作为 loss 进行反向传播更新参数，来达到 Self-Supervised Learning 的效果。

![image](https://github.com/icey-zhang/notebook/assets/54712081/fe5f2e09-1006-44e3-bf7f-c633e36fdbda)

**问：真实 patches 对应的 visual token 是怎么得到的呢？**

答：如3.2.2节介绍。训练一个 dVAE，其中的 Tokenizer 的作用就是把 image patches 编码成 visual tokens，通过 Tokenizer 来实现。

![image](https://github.com/icey-zhang/notebook/assets/54712081/15a2a4ce-b2ae-44be-88f4-6a0b6ee17435)
图16：BEIT的训练方法：对盖住的每个 patches，BEIT 的 Encoder 在这个位置的输出通过线性分类器之后得到预测的 visual token 与真实 patches 对应的 visual token 越接近越好
![image](https://github.com/icey-zhang/notebook/assets/54712081/94628916-4973-419d-b595-a0e0296fac92)
![image](https://github.com/icey-zhang/notebook/assets/54712081/4d938998-521b-40ce-8a5c-6b9fe5581d20)

下面的问题是如何随机盖住40% 的 image patches？

![image](https://github.com/icey-zhang/notebook/assets/54712081/44558a1d-64db-40cf-ab31-329c5806e5c7)
![image](https://github.com/icey-zhang/notebook/assets/54712081/f2fd3ab6-fe13-4f02-be35-cc20b3f787c0)
图17：Blockwise masking 的方法


3.2.5 BEIT 的目标函数：VAE 视角
![image](https://github.com/icey-zhang/notebook/assets/54712081/7181f40f-8e1a-422d-9204-c5c39d98026f)

![image](https://github.com/icey-zhang/notebook/assets/54712081/625837e4-5ee0-4db2-99b5-b87c0ceebe81)
上式11就是 BEIT 的总目标函数，使用 Gradient Ascent 更新参数。

所以，BEIT 遵循 BERT 的训练方法，让 BEIT 看很多的图片，随机盖住一些 image patches，让 BEIT 模型预测盖住的 patches 是什么，不断计算预测的 patches 与真实的 patches 之间的差异，利用 12 式进行反向传播更新参数，来达到 Self-Supervised Learning 的效果。

不同的是，BERT 的 Encoder 输入是 token，输出还是 token，让盖住的 token 与输出的预测 token 越接近越好；**而 BEIT 的 Encoder 输入是 image patches，输出是 visual tokens，让盖住的位置输出的 visual tokens 与真实的 visual tokens 越接近越好。真实的 visual tokens 是通过一个额外训练的 dVAE 得到的。**
BEIT Encoder 的具体架构细节：12层 Transformer，Embedding dimension=768，heads=12，FFN expansion ratio=4，Patch Size=16，visual token总数，即词汇表大小 $|V|=8192$。Mask 75个 patches，一个196个，大约占了40%。

BEIT Encoder 的具体训练细节：在 ImageNet-1K上预训练。

![image](https://github.com/icey-zhang/notebook/assets/54712081/b9548fd1-9efc-40be-987a-b4d94ba13ef8)
图18： BEIT 在下游分类任务 Fine-tuning

3.2.8 实验
分类实验

BEIT 实验的具体做法遵循3.2.7节的BEIT 在下游任务 Fine-tuning的做法，展示的都是预训练模型在具体小数据集上面 Fine-tune之后得到的结果。分类实验在CIFAR-10和ImageNet这两个数据集上进行，超参数设置如下图19所示：

![image](https://github.com/icey-zhang/notebook/assets/54712081/2d05e517-3ef1-42c0-a60b-4691f0498409)
图19：CIFAR-10和ImageNet超参数

下图20是实验在CIFAR-10和ImageNet这两个数据集上的性能以及与其他模型的对比。所有的模型大小都是 "base" 级别。 与随机初始化训练的模型相比，作者发现预训练的BEIT模型在两种数据集上的性能都有显著提高。值得注意的是，在较小的CIFAR-100数据集上，从头训练的ViT仅达到48.5%的准确率。相比之下，通过Pre-train的帮助，BEIT达到了90.1%。结果表明，BEIT可以大大降低有标签数据 (labeled data) 的需求。BEIT还提高了ImageNet上的性能。


![image](https://github.com/icey-zhang/notebook/assets/54712081/4ba637cc-62f1-4b7c-9c0b-45f69dde0de0)
图20：BEIT在CIFAR-10和ImageNet这两个数据集上的性能以及与其他模型的对比

此外，作者将BEIT与21年几个最先进的 Transformer 自监督方法进行比较，如 DINO 和 MoCo v3 (这2个模型也会在这个系列中解读)。我们提出的方法在ImageNet微调上优于以往的模型。BEIT在ImageNet上的表现优于DINO，在CIFAR-100上优于MoCo v3。此外，作者评估了我们提出的方法与 Intermediate Fine-tuning。换句话说，我们首先以自监督的方式对BEIT 进行预训练，然后用标记数据在 ImageNet 上对预训练的模型进行 Fine-tune。结果表明，在ImageNet上进行 Intermediate Fine-tuning 后获得额外的增益。

问：图20中的 Supervised Pre-Training on ImageNet 和 Supervised Pre-Training, and Intermediate Fine-tuning on ImageNet有什么区别？

答：二者都是使用全部的 ImageNet-1K 数据集。前者是只训练分类器的参数，而 BEIT 预训练模型参数不变。后者是既训练分类器的参数，又微调 BEIT 预训练模型参数。



作者也在 384×384 高分辨率数据集上面作 Fine-tune 了 10个epochs，同时patch的大小保持不变，也就是用了序列长度增加了。 结果如下图21所示，在ImageNet上，更高的分辨率可以提高1个点的。更重要的是，当使用相同的输入分辨率时，用 ImageNet-1K 进行预训练的BEIT-384 甚至比使用 ImageNet-22K 进行监督预训练的 ViT-384 表现更好。


![image](https://github.com/icey-zhang/notebook/assets/54712081/502633d5-5c5f-48d0-a1e4-32cc345208ab)
图21：Top-1 accuracy on ImageNet-1K

作者进一步扩大了 BEIT 的规模 (扩大到与 ViT-L 相同)。如上图21所示，在ImageNet上，从头开始训练时，ViT-384-L 比 ViT-384差。结果验证了 Vision Transformer 模型的 data hungry 的问题。解决方法就是用更大的数据集 ImageNet-22K，用了以后 ViT-384-L 最终比ViT-384 涨了1.2个点。相比之下，BEIT-L比 BEIT 好2个点，BEIT-384-L 比 BEIT-384 好1.7个点，说明大数据集对BEIT的帮助更大。



对比实验：

消融实验分别是在ImageNet (分类) 和 ADE20K (分割) 任务上进行的，自监督方式训练 epochs是300。

第1个探索Blockwise masking的作用。Blockwise masking 指的是图17的方法，发现它在两种任务中都是有利的，特别是在语义分割上。

第2个探索 recover masked pixels的作用，recover masked pixels指的是盖住一个 image patch，BEIT 的 Encoder 模型不输出visual token，而是直接进行 pixel level的回归任务，就是直接输出这个 patch，发现这样也是可以的，只是精度稍微变差了。这说明预测 visual tokens 而不是直接进行 pixel level的回归任务才是 BEIT 的关键。

第3个探索 1,2 的结合方案，去掉Blockwise masking，以及直接进行 pixel level的回归任务，这个性能是最差的。

第4个探索不进行自监督预训练，即直接恢复100%的image patches，性能也会下降。

![image](https://github.com/icey-zhang/notebook/assets/54712081/9aacd223-b1f5-4b35-b812-3dbe15b5cf08)

下图23是BEIT模型不同reference points的attention map，可视化的方法是拿出BEIT的最后一个layer，假定一个参考点，随机选定它所在的patch，比如是第57个patch，然后把attention map的第57行拿出来，代表这个第57号patch attend to所有patch的程度，再reshape成正方形就得到了下图23。

可以发现仅仅是预训练完以后，BEIT 就能够使用 self-attention 来区分不同的语义区域。 这个性质表明了为什么 BEIT 能够帮助下游任务的原因。通过BEIT获得的这些知识有可能提高微调模型的泛化能力，特别是在小数据集上。

![image](https://github.com/icey-zhang/notebook/assets/54712081/d051a88e-c0de-400c-b12d-71f3b51c5b13)
![image](https://github.com/icey-zhang/notebook/assets/54712081/12fee7ad-c3fe-43e1-bf83-0a7f281949ce)
图23：不同reference points的attention map

总结：
BEIT 遵循 BERT 的训练方法，让 BEIT 看很多的图片，随机盖住一些 image patches，让 BEIT 模型预测盖住的patches是什么，不断计算预测的 patches 与真实的 patches 之间的差异，利用它作为 loss 进行反向传播更新参数，来达到 Self-Supervised Learning 的效果。

不同的是，BERT 的 Encoder 输入是 token，输出还是 token，让盖住的 token 与输出的预测 token 越接近越好；而 BEIT 的 Encoder 输入是 image patches，输出是 visual tokens，让盖住的位置输出的 visual tokens 与真实的 visual tokens 越接近越好。真实的 visual tokens 是通过一个额外训练的 dVAE 得到的。

2021年12月1号

参考：
[变分自编码器](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/5253)
[BEiT：视觉BERT预训练模型](https://zhuanlan.zhihu.com/p/381345343)
