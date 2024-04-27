# Mamba: Linear-Time Sequence Modeling with Selective State Spaces
基础模型现在为深度学习中的大多数令人兴奋的应用提供支持，几乎普遍基于 Transformer 架构及其核心注意模块。已经开发了许多次二次时间架构，**例如线性注意力、门控卷积和循环模型和结构化状态空间模型 (SSM)**，以解决 Transformer **在长序列上的计算效率低下**，但它们尚未执行以及对语言等重要模式的关注。我们发现此类模型的一个关键弱点是它们**无法执行基于内容的推理**，并进行了一些改进。首先，简单地让 SSM 参数作为输入的函数解决了它们在**离散模态的弱点**，允许模型根据当前标记沿序列长度维度选择性地传播或忘记信息。其次，即使这种变化阻止了高效卷积的使用，我们在循环模式下设计了一种硬件感知的并行算法。我们将这些选择性 SSM 集成到一个简化的端到端神经网络架构中，无需注意甚至 MLP 块 (Mamba)。Mamba 具有快速推理（比 Transformer 高 5 倍）和序列长度的线性缩放，其性能在高达百万长序列的真实数据上有所提高。作为通用序列模型主干，Mamba 在语言、音频和基因组学等多种模式上实现了最先进的性能。在语言建模方面，我们的 Mamba-3B 模型在预训练和下游评估中都优于相同大小的 Transformer，并匹配 Transformer 的大小的两倍。

背景：S4架构
Mamba的架构主要基于S4，一种最新的状态空间模型（SSM，state space model）架构。
![image](https://github.com/icey-zhang/notebook/assets/54712081/42e4eeff-86b6-4f85-9fa2-111c1cdd1757)

在较高层次上，S4学习如何通过中间状态 h(t) 将输入x(t) 映射到输出 y(t) 上。

在此，由于SSM被设计于很好地处理连续数据，例如音频、传感器数据和图像，因此x、y、t 是x的函数。

S4通过三个连续参数矩阵A、B和C将它们互联，具体形式表现为以下两个方程（Mamba论文中的1a和1b）：
![image](https://github.com/icey-zhang/notebook/assets/54712081/8886cf36-ca21-453a-91f9-9471b5a18bff)

由于在实践中，我们一般都是处理离散数据比如文本，这就需要我们对SSM进行离散化，通过使用特殊的第四个参数Δ，将连续参数A、B和C转换为离散参数。

离散化后，我们可以通过这两个方程（Mamba论文中的2a和2b）来表示SSM：

![image](https://github.com/icey-zhang/notebook/assets/54712081/a0bfb720-8719-4bcc-babf-fc4e81d2f067)

这些方程形成一个递归，情况类似于咱在RNN网络中看到的一样。在每个步骤t中，我们将前一个时间步ht−1的隐藏状态与当前输入xt相结合，以创建新的隐藏状态ht。

下图展示了它在预测句子中的下一个单词时是如何工作的（我们预测“and”跟在“My name is Jack”之后）。

![image](https://github.com/icey-zhang/notebook/assets/54712081/31fc27f7-02e5-44f0-a21f-02db9335444c)

依据以此，我们本质上就可以使用S4作为递归神经网RNN来一次生成一个 token。

然而，S4真正酷的地方在于，你也可以将它用作卷积神经网络CNN。

在上面的示例中，当我们扩展之前的离散方程来尝试计算h3时，会发生什么？

为了简单起见，我们假设x−1=0。

![image](https://github.com/icey-zhang/notebook/assets/54712081/5be4d9c0-e2c4-4de2-8183-6925ea3d207e)

计算出h3后，我们可以将其代入y3的等式中来预测下一个单词：

![image](https://github.com/icey-zhang/notebook/assets/54712081/690f64cc-1fc7-465b-9385-ff1fcd04187b)

现在，请注意y3实际上可以计算为点积，其中右侧向量是我们的输入x：

![image](https://github.com/icey-zhang/notebook/assets/54712081/da058a12-c4e8-464b-87de-c865b86b7143)

由于其中三个离散参数A、B和C都是常数，因此我们可以预先计算左侧向量并将其保存为卷积核。这为我们提供了一种使用卷积计算y的简单方法，如以下两个方程所示（Mamba论文中的3a和3b）：

![image](https://github.com/icey-zhang/notebook/assets/54712081/990c1b39-1a63-4205-9217-ee07fea0050b)

划重点：这些循环和卷积形式（作者称之为“RNN模式”和“CNN模式”）在数学上是等效的。

因此S4可以根据你需要它执行的操作进行变形，同时输出没有任何差异。

当然，CNN模式更适合训练，RNN模式更适合推理。

第一个主要思想：可选性

这部分我们讨论Mamba引入的第一个主要思想：可选性。让我们回想一下定义S4离散形式的两个方程：

![image](https://github.com/icey-zhang/notebook/assets/54712081/86298f9f-8b93-4c73-a4df-7e9e9afbe635)

注意，在S4中，我们的离散参数AB和C是恒定的。然而，Mamba使这些参数根据输入而变化。因此我们最终会得到这样的结果：

![image](https://github.com/icey-zhang/notebook/assets/54712081/56671be7-ed49-485b-9bdd-d8effc8cca89)

Mamba作者（Gu和Dao）认为，选择性或输入依赖性对于许多任务都很重要。

而本文的科普作者则认为：因为S4没有选择性，所以它被迫以完全相同的方式处理输入的所有部分。

然而，当我们面对一句话时，其中有些单词不可避免地比其他单词更重要。

就比如 “I want to order a hamburger.”这句。

如果没有选择性，S4会花费相同的“精力”来处理每个单词：

![image](https://github.com/icey-zhang/notebook/assets/54712081/08fcd197-8cdd-4bcd-ab5d-1aa536436230)


但如果是一个试图对这句话的意图进行分类的模型，它可能会想更多地“关注”order、hamburger，而不是want、to。

如下图所示，而通过使模型参数成为输入的函数，Mamba就可以做到“专注于”输入中对于当前任务更重要的部分。

![image](https://github.com/icey-zhang/notebook/assets/54712081/a8b8b8da-15bc-4cca-beac-462c01723f39)

然而，选择性给我们带来了一个问题。让我们回想一下之前计算的卷积核。

![image](https://github.com/icey-zhang/notebook/assets/54712081/de9924e7-50b0-4a69-b5ed-8ce74776ec8e)

在S4中，我们可以预先计算该内核、保存，并将其与输入x相乘。

这很好，因为离散参数AB和C是恒定的。**但同样，在Mamba中，这些矩阵会根据输入而变化！因此，我们无法预计算K，也无法使用CNN模式来训练我们的模型。如果我们想要选择性，我们得用RNN模式进行训练。**方法是删除方程3b以获得“戏剧性的效果”。

![image](https://github.com/icey-zhang/notebook/assets/54712081/278657c2-e73f-4f99-9e6d-759f6686a61e)

但这给Mamba的作者带来了一个问题：**RNN模式的训练速度非常慢。**

假如我们正在使用1000个token的序列训练我们的模型：

CNN本质上会计算其内核和输入向量之间的点积，并且可以并行执行这些计算。相比之下，RNN需要按顺序更新其隐藏状态1000次。

这便导致Mamba的作者提出了他们的第二个伟大思想。

第二个主要思想：无需卷积的快速训练
Mamba可以在RNN模式下进行非常非常快速的训练。

在某个时刻，它们的递归与扫描算法（也称为前缀和，prefix sum）非常相似。

要计算前缀和，我们需要获取一个输入数组 [x1，x2，… ，xn] ，并返回一个输出数组，其中每个元素都是该项目及其之前项目的总和。

换句话说，输出的第一个元素将为x1 ，第二个元素将为[x1+[x2 ，依此类推。一个例子：

![image](https://github.com/icey-zhang/notebook/assets/54712081/dd292cd1-4cae-47fb-889d-cd0e71476a5b)


现在我们画出RNN模式下更新Mamba隐藏状态的流程。

![image](https://github.com/icey-zhang/notebook/assets/54712081/e30cf5f6-ef25-4392-ac6c-7fff6f63296e)


等等……，如果我们必须形式化前缀和，我们可以将其写成以下等式：

![image](https://github.com/icey-zhang/notebook/assets/54712081/478f6528-fdc9-4761-83ef-a86a1c85464a)


该方程形成一个递归：在每一步，我们通过将先前存储的值添加到当前输入来计算新值。现在，让我们再次看看更新之后Mamba隐藏状态的循环。

![image](https://github.com/icey-zhang/notebook/assets/54712081/9efad322-4c08-47ff-a9ce-cc73ea4d59de)


这两个等式真的非常非常相似有么有！

而最酷的地方又来了：虽然计算前缀和本质上看起来似乎是顺序的，但我们实际上拥有用于此任务的高效并行算法！

在下图中，我们可以看到正在运行的并行前缀和算法，其中每条垂直线代表数组中的一项。

![image](https://github.com/icey-zhang/notebook/assets/54712081/dcee1e92-736f-4a7f-860a-65acede36c17)


花一点时间捋一下这个算法：

选择任何垂直线，从顶部开始，然后向下移动，将每个加法追溯到数组的前几个项目。当你到达底部时，应该在行的左侧看到所有项目的总和。

例如，在第一个元素添加到开头的第二个元素之后，数组的第三个元素在末尾接收了第二个元素的添加值。结果，当并行扫描完成时，第三个元素包含第一、第二和第三元素的总和。

如果我们在没有并行性的单线程中运行该算法，则比仅按顺序将值相加所需的时间要长。但GPU拥有大量处理器，可以进行高度并行计算。因此，我们可以在大约O(logn) 时间内计算此前缀和（或扫描）操作！

因此，Mamba的作者意识到，如果他们想在RNN模式下高效训练，他们可能可以用并行扫描。

但由于PyTorch目前没有扫描实现，Mamba的作者自己编写了一个——但，结果并不好。

![image](https://github.com/icey-zhang/notebook/assets/54712081/a4147293-85a5-49d4-a617-4be5a54de6ad)


在上图中，大家可以看到他们基于PyTorch的扫描实现（绿色）总是慢于FlashAttention-2（蓝色），FlashAttention-2是可用“精确注意力”的最快实现。

尽管当序列长度为128000个token时，扫描似乎在运行时赶上，但还是耗尽了内存。

为了让Mamba变得实用，它需要更快。这让Mamba的作者看到了Dao之前关于FlashAttention的工作，从而解决了问题。

由于篇幅所限，在此我们省略了原文中FlashAttention的原理介绍部分（Review: FlashAttention），感兴趣的朋友可以查看原博/FlashAttention原论文，或者我们之前的一篇[原理介绍文章](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzIzNjc1NzUzMw%3D%3D%26mid%3D2247686895%26idx%3D3%26sn%3D18e1c3fd5db81cf3c4a72c223b6344f6%26chksm%3De8dead9ddfa9248b30ed2f3f961659ff84c1e7d47aa35a5fa78e9c5c8f28fc61269935c47fc3%26scene%3D21%23wechat_redirect)。

Back to Mamba
还是基于上一张对比图。

事实证明，如果在计算扫描时采用相同的内存感知平铺方法，则可以大大加快速度。

通过这种优化，Mamba（红色）现在在所有序列长度上都比 FlashAttention-2（蓝色）更快。

![image](https://github.com/icey-zhang/notebook/assets/54712081/f283ef4b-d3c5-4b50-aa78-9789f8df8a16)


这些结果表明，就速度而言，Mamba是实用的，其运行速度比最快的Transformer还要快。但它在语言建模方面有什么擅长的地方吗？

Mamba作者在涉及语言、基因组学和音频的许多序列建模任务上对Mamba进行了评估。

结果看起来很酷：Mamba在对人类基因组项目的DNA和钢琴音乐数据集的音频进行建模时建立了最先进的性能。

然而，让很多人兴奋的是语言任务上的结果。许多关于Mamba的在线讨论都集中在下图中：

![image](https://github.com/icey-zhang/notebook/assets/54712081/9cdc4bec-39de-420f-aaac-ff3f759705ac)


我们可以看到，模型大小向右增加，语言建模性能则随着进一步向下而提高。

这意味着最好的模型应该位于左侧：体积小（因此速度快），并且非常擅长建模语言。

由于Mamba作者都是学者，搞不来数千个GPU来训练GPT-4大小的模型，因此实验是通过训练一堆较小的模型（大约125M到1.3B参数）来进行比较的。

如上图所示，结果看起来非常有希望。与其他类似尺寸的模型相比，Mamba似乎是最擅长建模语言的。

https://zhuanlan.zhihu.com/p/683978639

**https://blog.csdn.net/v_JULY_v/article/details/134923301**

## 第三部分 Mamba的三大创新
mamba(其对应论文为：Mamba: Linear-Time Sequence Modeling with Selective State Spaces，这是其对应的GitHub代码地址)，在语言、音频、DNA序列模态上都实现SOTA，在最受关注的语言任务上，Mamba-3B超越同等规模的Transformer，与两倍大的Transformer匹敌，并且相关代码、预训练模型checkpoint都已开源

简言之，Mamba是一种状态空间模型(SSM)，建立在更现代的适用于深度学习的结构化SSM (简称S6)基础上，与经典架构RNN有相似之处

### 3.1 Mamba = 有选择处理信息 + 硬件感知算法 + 更简单的SSM架构
与先前的研究相比，Mamba主要有三点创新：

**1.对输入信息有选择性处理(Selection Mechanism)**

**2.硬件感知的算法(Hardware-aware Algorithm)**
该算法采用“并行扫描算法”而非“卷积”来进行模型的循环计算(使得不用CNN也能并行训练)，但为了减少GPU内存层次结构中不同级别之间的IO访问，它没有具体化扩展状态
当然，这点也是受到了S5(Simplified State Space Layers for Sequence Modeling)的启发

**3.更简单的架构**
将SSM架构的设计与transformer的MLP块合并为一个块(combining the design of prior SSM architectures with the MLP block of Transformers into a single block)，来简化过去的深度序列模型架构，从而得到一个包含selective state space的架构设计

#### 3.1.1 选择性状态空间模型：从S4到S6
作者认为，序列建模的一个基础问题是把上下文压缩成更小的状态(We argue that a fundamental problem of sequence modeling is compressing context into a smaller state)，从这个角度来看

- transformer的注意力机制虽然有效果但效率不算很高，毕竟其需要显式地存储整个上下文(storing the entire context，也就是KV缓存)，直接导致训练和推理消耗算力大
好比，Transformer就像人类每写一个字之前，都把前面的所有字+输入都复习一遍，所以写的慢
- RNN的推理和训练效率高，但性能容易受到对上下文压缩程度的限制
On the other hand, recurrent models are efficient because they have a finite state, implying constant-time inference and linear-time training. However, their effectiveness is limited by how well this state has compressed the context.

好比，RNN每次只参考前面固定的字数(仔细体会这句话：When generating the output, the RNN only needs to consider the previous hidden state and current input. It prevents recalculating all previous hidden states which is what a Transformer would do)，写的快是快，但容易忘掉更前面的内容

- 而SSM的问题在于其中的矩阵A B C不随输入不同而不同，即无法针对不同的输入针对性的推理，详见上文的2.4节

![image](https://github.com/icey-zhang/notebook/assets/54712081/13c84216-0b2c-43f6-8a75-3fc56470d9cb)

- 最终，Mamba的解决办法是，相比SSM压缩所有历史记录，mamba设计了一个简单的选择机制，通过“参数化SSM的输入”，让模型对信息有选择性处理，以便关注或忽略特定的输入
这样一来，模型能够过滤掉与问题无关的信息，并且可以长期记住与问题相关的信息
好比，Mamba每次参考前面所有内容的一个概括，越往后写对前面内容概括得越狠，丢掉细节、保留大意

为方便大家对比，我再用如下表格总结下各个模型的核心特点
![image](https://github.com/icey-zhang/notebook/assets/54712081/3dd67973-84fa-4e6d-aa89-8da6f81ab74b)

总之，序列模型的效率与效果的权衡点在于它们对状态的压缩程度：

- 高效的模型必须有一个小的状态(比如RNN或S4)
- 而有效的模型必须有一个包含来自上下文的所有必要信息的状态(比如transformer)

而mamba为了兼顾效率和效果，选择性的关注必须关注的、过滤掉可以忽略的

![image](https://github.com/icey-zhang/notebook/assets/54712081/5beeed44-c4de-4d7b-b7b1-56ec28e4c72b)

##### 3.1.1.1 mamba前身S4的4个参数的不随输入不同而不同
首先，在其前身S4中，其有4个参数(∆, A, B, C)

![image](https://github.com/icey-zhang/notebook/assets/54712081/880af0f4-6016-444d-819e-81fadbc59ac1)

且它们不随输入变化(即与输入无关)，这些参数控制了以下两个阶段


![image](https://github.com/icey-zhang/notebook/assets/54712081/ee230256-71cd-457b-b256-42baa722c8b0)


![image](https://github.com/icey-zhang/notebook/assets/54712081/d2e1dc0e-d8c3-4e94-8b98-ce259a87ecfe)


##### 3.1.1.2 S4中三个矩阵的维度表示、维度变化
![image](https://github.com/icey-zhang/notebook/assets/54712081/23865dee-f867-4141-b845-ec03d68af06b)


![image](https://github.com/icey-zhang/notebook/assets/54712081/e417e85f-ec66-4e84-8c07-abadd528442e)

1.但为了对批量大小为B、长度为L(注意，N <<L，比如类似上文举的例子中，N = 64 L=10000)、具有D个通道(虽然在之前的示例中，每个token的维度设定的1，比如拿 一个 64 × 64维的矩阵A 去记 10000 × 1维的数字，但实际上，经常会遇到一个token不止一个维度的，比如颜色便有R G B三个通道，即embedding的dimension是D )的输入序列x进行操作「总之，x,y则是输入和输出，和 Transformer 里面一样, 他们的大小是 (batch size B x sequence length L x embedding dim D)」

![image](https://github.com/icey-zhang/notebook/assets/54712081/ee803c22-28b7-478a-b777-cd952fb631b2)


Mamba 的处理方式是，给这 D 个 dimension的每个 dimension 都搞一个独立的 SSM，即SSM被独立地应用于每个通道(To operate over an input sequence 𝑥 of batch size 𝐵 and length 𝐿 with 𝐷 channels, the SSM is applied independently to each channel)

2.这就解释了为什么下图中的A、B、C三个矩阵的第一个维度是都是 D

![image](https://github.com/icey-zhang/notebook/assets/54712081/f7776191-ea10-4ffe-889f-12031c31c817)


请注意，在这种情况下，每个输入的总隐藏状态具有DN维，在序列长度上计算它需要O(BLDN)的时间和内存(the total hidden state has dimension 𝐷𝑁 per input, and computing it over the sequence length requires 𝑂(𝐵𝐿𝐷𝑁) time and memory)

3.1.1.3 mamba：从S4到S6的算法变化流程
最后，在Mamaba中，作者让B矩阵、C矩阵、∆成为输入的函数，让模型能够根据输入内容自适应地调整其行为

![image](https://github.com/icey-zhang/notebook/assets/54712081/e97a475b-5198-47ab-bedb-8ddf4765e51e)

1.从S4到S6的过程中
\rightarrow  影响输入的B矩阵、影响状态的C矩阵的大小从原来的(D,N)「前面说了，D指的是输入向量的维度，比如一个颜色的变量一般有R G B三个维度，N指SSM的隐藏层维度hidden dimension，当然 一般设的比较小，远小于L 」

![image](https://github.com/icey-zhang/notebook/assets/54712081/52d803b2-6ee9-4e69-a41f-f7c1f09dae1d)

变成了(B,L,N)「这三个参数分别对应batch size、sequence length、hidden state size」

![image](https://github.com/icey-zhang/notebook/assets/54712081/c71fd8d3-e0ac-4e8c-b859-fa5c4d9aacd7)

\rightarrow 且的大小由原来的D变成了(B,L,D)，意味着对于一个 batch 里的 每个 token (总共有 BxL 个)都有一个独特的
且每个位置的矩阵、矩阵、都不相同，这意味着对于每个输入token，现在都有独特不同的矩阵、矩阵，可以解决内容感知问题

2.维度上的变化具体执行时是怎么实现的呢？好办，通过

![image](https://github.com/icey-zhang/notebook/assets/54712081/b120a893-faf3-4e83-8def-d3abb2bad8fb)


