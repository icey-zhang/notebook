深度解析 [Transformer](https://so.csdn.net/so/search?q=Transformer&spm=1001.2101.3001.7020) 和注意力机制
----------------------------------------------------------------------------------------------

在[《图解NLP模型发展：从RNN到Transformer》](https://jarod.blog.csdn.net/article/details/129564388)一文中，我介绍了 [NLP](https://so.csdn.net/so/search?q=NLP&spm=1001.2101.3001.7020) 模型的发展演化历程，并用直观图解的方式为大家展现了各技术的架构和不足。有读者反馈图解方式虽然直观，但深度不足。考虑到 Transformer 是大模型的基石，本文将重点为大家深入剖析 Transformer 和注意力机制。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a834a23ce9b548aea19c780776dce9d3.png#pic_center)

图1\. Transformer之后大语言模型的发展

这是将是我迄今为止最长的文章，几乎涵盖了关于 Transformer 和注意力机制的所有必要内容，包括自注意力、查询、键、值、多头注意力、掩码多头注意力和 Transformer 架构，以及完整的PyTorch代码实现。希望阅读完本文大家对 Transformer 能有深入的理解。

#### 文章目录

*   *   [Transformer 的诞生背景和意义](#Transformer__15)
    *   *   [RNN的问题](#RNN_19)
    *   [注意力机制](#_39)
    *   *   [词嵌入](#_41)
        *   [自注意力](#_49)
        *   [查询, 键, 值](#___112)
        *   [注意力](#_146)
        *   [注意力机制的神经网络表示](#_220)
        *   [掩码注意力](#_229)
        *   [多头注意力](#_250)
    *   [Transformer 网络](#Transformer__277)
    *   *   [编码器](#_293)
        *   *   [输入嵌入](#_302)
            *   [位置编码](#_311)
            *   [多头注意力](#_336)
            *   [Add & Norm 与前馈](#Add__Norm__361)
        *   [解码器](#_371)
    *   [用PyTorch实现Transformer](#PyTorchTransformer_399)
    *   *   [导入必要的库和模块](#_412)
        *   [定义基础模块](#_425)
        *   *   [多头注意力](#_429)
            *   [位置前馈网络](#_476)
            *   [位置编码](#_492)
        *   [编码器层](#_518)
        *   [解码器层](#_542)
        *   [Transformer模型](#Transformer_579)
        *   [准备样本数据](#_634)
        *   [训练模型](#_655)
    *   [总结](#_676)

### Transformer 的诞生背景和意义

我们还是先从 Transformer 和注意力机制的历史开始讲起。其实注意力机制比 Transformer 出现地更早。注意力机制在 2014 年就首次用于计算机视觉领域，试图理解神经网络进行预测时正在观察什么。这是尝试理解卷积神经网络 (CNN) 输出的第一步。2015 年，注意力机制首次出现在自然语言处理（NLP）领域，用于对齐机器翻译。 最后，在 2017 年，注意力机制被加入到 Transformer 网络，用于语言建模。 此后，Transformers 超越了 [RNN](https://so.csdn.net/so/search?q=RNN&spm=1001.2101.3001.7020) 的预测精度，成为 NLP 领域最先进地技术。

#### RNN的问题

Transformers 的出现取代了 RNN 在 NLP 领域的地位。归根结底是因为 RNN 存在一些问题而 Transformer 解决了这些问题。

**问题1\. 长程依赖问题**

RNN 存在长程依赖问题，不适用于长文本。而 Transformer 网络几乎只使用注意力模块。注意力有助于在序列的任何部分之间建立联系，因此不存在长程依赖问题。 对于 Transformer 而言，长程依赖与短程依赖的处理方式是一样的。

**问题2\. 梯度消失和梯度爆炸**

RNN 饱受梯度消失和梯度爆炸之苦。而 Transformer 几乎没有梯度消失或梯度爆炸问题。在 Transformer 网络中，整个序列是同时训练的，因此很少有梯度消失或梯度爆炸问题。

**问题3\. 训练性能低**

RNN 需要更多的训练步骤才能达到局部/全局最优。 我们可以将 RNN 视为非常深的展开网络，网络的大小取决于序列的长度。这将产生许多参数，并且这些参数中的大部分是相互关联的。 这就导致优化需要更长的训练时间和更多的步骤。而 Transformer 需要的训练步骤比 RNN 要少。

**问题4\. 无法并行**

RNN 无法并行计算。因为 RNN 是序列模型，即网络中的所有计算都是顺序发生的，每一步操作都依赖前一步的输出，因此很难并行化。而 Transformer 网络允许并行计算，可以充分发挥 GPU 并行计算优势。

### 注意力机制

#### 词嵌入

计算机是很难直接使用自然语言文本的，因此在 NLP 中，第一步都是需要将自然语言单词转换为向量。将文本中的单词转换等长向量就是**嵌入**。 嵌入中的每个维度都具有潜在的意义。例如，第一个维度可以表征单词的“阳刚之气”。 第一维中的数字越大，该词与男性相关的可能性就越大。这里仅仅是为了方便大家理解的一个举例，在具体实践中，很难暴露向量维度的含义。

词嵌入没有一个普遍标准。同一个词的嵌入因各种神经网络而异，也会因训练阶段而异。嵌入从随机值开始，并在训练期间不断调整以最小化神经网络误差。

将句子中每个单词的嵌入集合在一起，就得到**嵌入矩阵**，矩阵中每一行代表一个词嵌入。

#### 自注意力

举个例子，比如下面这句话：  
小美长得很漂亮而且人还很好 \\text{小美长得很漂亮而且人还很好} 小美长得很漂亮而且人还很好  
如果我们单看句中**“人”**这个字，会发现**“而且”**和**“还”**是距离它最近的两个词，但这两个词并没有带来任何上下文信息，反而**“小美”**和**“好”**这两个词与**“人”**的关系更密切。这句话后半部分的意思是“小美人好”。这个例子告诉我们——词语位置上的接近度并不总是与意义相关，上下文更重要。

当这个句子被输入到计算机时，程序会将每个词视为一个token t t t，每个token都有一个词嵌入 A A A。但是这些词嵌入没有上下文。所以注意力机制的思想是应用某种权重或相似性，让初始词嵌入 A A A 获得更多上下文信息，从而获得最终带上下文的词嵌入 Y Y Y。

![在这里插入图片描述](https://img-blog.csdnimg.cn/998ff67fa13f4498a5262b7f56d9353b.png#pic_center)

图2\. 自注意力实例解释

在嵌入空间中，相似的词出现得更近或具有相似的嵌入。 例如“程序员”这个词与“代码”和“开发”的关系比与“口红”的关系更大。同样，“口红”与“眼影”、“粉底”的关系比与“火箭”一词的关系更大。

所以，直觉上，如果“程序员”这个词出现在句子的开头，而“代码”这个词出现在句子的结尾，它们应该为彼此提供更好的上下文。我们用这一思想来找到权重向量 W W W，通过将词嵌入相乘（点积）以获得更多上下文。 因此，在句子 _“小美长得很漂亮而且人还很好”_ 中，我们没有按原样使用词嵌入，而是将每个词的嵌入相互相乘。 下面计算公式演示可以更好地说明这一点。

1.  发现权重

{ a 1 a 1 = w 11 a 1 a 2 = w 13 a 1 a 3 = w 13 ⋮ a 1 a n = w 1 n n o r m a l i z e → w 11 w 13 w 13 ⋮ w 1 n } 重新计算第一个向量的权重 (1)

⎧⎩⎨⎪⎪⎪⎪a1a1=w11a1a2=w13a1a3=w13⋮a1an=w1n{a1a1=w11a1a2=w13a1a3=w13⋮a1an=w1n

\\begin{cases} a\_1 a\_1 = w_{11} \\\ a\_1 a\_2 = w_{13} \\\ a\_1 a\_3 = w_{13} \\\ \\vdots \\\ a\_1 a\_n = w_{1n} \\end{cases} \\qquad\\underrightarrow{normalize}\\qquad

\\begin{rcases} w_{11} \\\ w_{13} \\\ w_{13} \\\ \\vdots \\\ w_{1n} \\end{rcases}\\begin{rcases} w_{11} \\\ w_{13} \\\ w_{13} \\\ \\vdots \\\ w_{1n} \\end{rcases}

\\begin{rcases} w_{11} \\\ w_{13} \\\ w_{13} \\\ \\vdots \\\ w_{1n} \\end{rcases} \\quad \\text{重新计算第一个向量的权重} \\tag{1} ⎩  ⎨  ⎧​a1​a1​=w11​a1​a2​=w13​a1​a3​=w13​⋮a1​an​=w1n​​ normalize​w11​w13​w13​⋮w1n​​⎭  ⎬  ⎫​重新计算第一个向量的权重(1)

2.  获取带上下文的词嵌入

w 11 a 1 + w 12 a 2 + w 13 a 3 + ⋯ + w 1 n a n = y 1 w 21 a 1 + w 22 a 2 + w 23 a 3 + ⋯ + w 2 n a n = y 2 ⋮ w n 1 a 1 + w n 2 a 2 + w n 3 a 3 + ⋯ + w n n a n = y n (2) w_{11}a\_1+w\_{12}a\_2+w\_{13}a\_3+\\dots+w\_{1n}a\_n = y\_1 \\\ w_{21}a\_1+w\_{22}a\_2+w\_{23}a\_3+\\dots+w\_{2n}a\_n = y\_2 \\\ \\vdots \\\ w_{n1}a\_1+w\_{n2}a\_2+w\_{n3}a\_3+\\dots+w\_{nn}a\_n = y\_n \\\ \\tag{2} w11​a1​+w12​a2​+w13​a3​+⋯+w1n​an​=y1​w21​a1​+w22​a2​+w23​a3​+⋯+w2n​an​=y2​⋮wn1​a1​+wn2​a2​+wn3​a3​+⋯+wnn​an​=yn​(2)

正如上面计算公式所示，我们首先将第一个词的初始嵌入与句子中所有其他词的嵌入相乘（点积）来找到新一组权重。这组权重（ w 11 w_{11} w11​ 到 w 1 n w_{1n} w1n​）会被归一化处理（一般使用 `softmax`）。接着，这组权重与句子中所有单词的初始嵌入相乘  
w 11 a 1 + w 12 a 2 + w 13 a 3 + ⋯ + w 1 n a n = y 1 (3) w_{11}a\_1+w\_{12}a\_2+w\_{13}a\_3+\\dots+w\_{1n}a\_n = y\_1 \\tag{3} w11​a1​+w12​a2​+w13​a3​+⋯+w1n​an​=y1​(3)  
w 11 w_{11} w11​ 到 w 1 n w_{1n} w1n​ 记录第一个词 a 1 a_1 a1​ 上下文的权重。因此，当我们将这些权重乘以每个词时，我们实际上是在将所有其他词重新加权到第一个词。所以从某种意义上说，“小美”这个词现在更倾向于“漂亮”和“好”，而不是紧随其后的词。 这在某种程度上提供了一定上下文信息。

对所有词重复此操作，便会让句子中每一个词从其他词上获得一定上下文信息。用向量形式来表示此过程会很简洁，见下面公式。  
softmax ( A ⋅ A T ) = W W ⋅ A = Y (4)

softmax(A\\sdotAT)W\\sdotA=W=Ysoftmax(A\\sdotAT)=WW\\sdotA=Y

\\begin{aligned} \\text{softmax}(A \\sdot A^T) &= W \\\ W \\sdot A &= Y \\end{aligned}\\tag{4} softmax(A⋅AT)W⋅A​=W=Y​(4)  
这里的权重不是训练得到的，并且词的顺序或接近程度相互之间没有影响。此外，该过程与句子的长度无关，也就是说，句子中词的多少无关紧要。这种为句子中的词添加上下文的方法被称为**自注意力**。

#### 查询, 键, 值

自注意力的问题在于没有训练任何东西。于是我们自然地会想到如果向其中添加一些可训练的参数，网络应该可以学习到一些模式，从而提供更好的上下文。于是便引入了 **查询(Query), 键(Key), 值(Value)** 的思想。

我们还是复用前面的例子——_“小美长得很漂亮而且人还很好”_ 。在自注意力公式中，我们发现初始词嵌入 V V V 出现了3次。前两次是作为句中词向量与其他词（包括它自己）点积得到权重；第三次再与权重相乘得到最终带上下文的词嵌入。这三个地方出现的词嵌入 A A A 我们给他们三个术语：**查询(Query), 键(Key), 值(Value)**。

假设我们想让所有的词都与第一个词 v 1 v_1 v1​ 相似。我们可以让 v 1 v_1 v1​ 作为查询。 然后，将该查询与句子中所有词（ v 1 v_1 v1​ 到 v n v_n vn​）进行点积，这里 v 1 v_1 v1​ 到 v n v_n vn​ 就是键。 所以查询和键的组合给了我们权重。接着再将这些权重与作为值的所有单词（ v 1 v_1 v1​ 到 v n v_n vn​）相乘。 这就是查询(Query)、键(Key)、值(Value)。下面的公式很好地指明了查询(Query)、键(Key)、值(Value)对应的部分。  
softmax ( A ⏟ Query ⋅ A T ⏟ Key ) = W W ⋅ A ⏟ Value = Y (5)

softmax(A⏟Query\\sdotAT⏟Key)W\\sdotA⏟Value=W=Ysoftmax(A⏟Query\\sdotAT⏟Key)=WW\\sdotA⏟Value=Y

\\begin{aligned} \\text{softmax}(\\underbrace{A}_{\\text{Query}} \\sdot \\underbrace{A^T}_{\\text{Key}}) &= W \\\ W \\sdot \\underbrace{A}_{\\text{Value}} &= Y \\end{aligned}\\tag{5} softmax(Query A​​⋅Key AT​​)W⋅Value A​​​=W=Y​(5)  
那么在哪里添加可训练参数矩阵呢？其实很简单。我们知道，如果一个 1 × k 1 \\times k 1×k 的向量乘以一个 k × k k \\times k k×k 的矩阵，结果是一个 1 × k 1 \\times k 1×k 的向量。如果我们将 A 1 A_1 A1​ 到 A n A_n An​ 中的每个键（每个 Key 的形状均为 1 × k 1 \\times k 1×k）与一个 k × k k \\times k k×k 的矩阵 W K W^K WK（Key 矩阵）相乘。同理，让查询向量与矩阵 W Q W^Q WQ（Query 矩阵）相乘，让值向量与矩阵 W V W^V WV （Value 矩阵）相乘，那么矩阵 W K W^K WK 、 W Q W^Q WQ 和 W V W^V WV 的值都可以通过神经网络进行训练，并提供比仅使用自注意力更好的上下文。

加入可训练参数矩阵后，我们的查询(Query)、键(Key)、值(Value)向量可以写作：  
Q = A W Q K = A W K V = A W V (6) Q = AW^Q\\\ K = AW^K\\\ V = AW^V\\\ \\tag{6} Q=AWQK=AWKV=AWV(6)  
代入公式（5）即可得到新的表达：  
softmax ( Q ⋅ K T ) = W W ⋅ V = Y (5)

softmax(Q\\sdotKT)W\\sdotV=W=Ysoftmax(Q\\sdotKT)=WW\\sdotV=Y

\\begin{aligned} \\text{softmax}(Q \\sdot K^T) &= W \\\ W \\sdot V &= Y \\end{aligned}\\tag{5} softmax(Q⋅KT)W⋅V​=W=Y​(5)  
将两部分连在一起，即可得到  
Y = softmax ( Q ⋅ K T ) ⋅ V (6) Y = \\text{softmax}(Q \\sdot K^T) \\sdot V \\tag{6} Y=softmax(Q⋅KT)⋅V(6)

#### 注意力

有了对查询(Query)、键(Key)、值(Value)的基本概念后，我们再来看一下注意力机制背后的官方步骤和公式。为了方便大家理解，我会通过一个数据库查询示例来解释注意力机制。

在数据库中，如果我们想通过查询 q q q 和键 k i k_i ki​ 检索某个值 v i v_i vi​，我们可以执行一些操作，使用查询来识别与特定值对应的键。下图显示了在数据库中检索数据的步骤。假设我们向数据库发送一个查询，通过某些操作可以找出数据库中的哪个键与查询最相似。一旦找到该键，则返回该键对应的值作为输出。在图中，该操作发现查询与 `Key 4` 最相似，因此将 `Key 4` 对应的值 `Value 4` 作为输出。

![在这里插入图片描述](https://img-blog.csdnimg.cn/0994410ba2b74b00bfe2e93f15207d53.png#pic_center)

图2\. 数据库取值过程

注意力与这种数据库取值技术类似，但是以概率的方式进行的。  
attension ( q , k , v ) = ∑ i similarity ( q , k i ) v i (7) \\text{attension}(q, k, v) = \\sum\_i \\text{similarity}(q, k\_i)v_i \\tag{7} attension(q,k,v)=i∑​similarity(q,ki​)vi​(7)

1.  注意力机制测量查询 q q q 和每个键值 k i k_i ki​ 之间的相似性。
2.  返回每个键值的权重代表这种相似性。
3.  最后，返回数据库中所有值的加权组合作为输出。

某种意义上，注意力与数据库检索的唯一区别是，在数据库检索中我们得到一个具体值作为输入，而在注意力机制中我们得到的是值的加权组合。例如，在注意力机制中，如果一个查询与 `Key 1` 和 `Key 4` 最相似，那么这两个 key 将获得最多的权重，输出将是 `Value 1` 和 `Value 4` 的组合。

下图展示了从查询、键和值中获得最终注意力值所需的步骤。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2aa9110d05bc427a8ad808b6ccf610d3.png#pic_center)

图3\. 获取注意力值的步骤

下面详细解释一下每个步骤。

**第一步**

第一步涉及键和查询以及相应的相似性度量。查询 q q q 会影响相似度。我们要做的就是通过查询和键计算出相似度，这里查询和键都是嵌入向量。相似度 S S S 被定义为查询 q q q 和键 k k k 的某种函数，可以使用多种方法计算，下面列举了一些常见的相似度计算函数：  
S i = f ( q , k ) = { q T ⋅ k i … … 点积 q T ⋅ k i / d … … 缩放点积 ( d 是键向量的维数 ) q T ⋅ W ⋅ k i … … 一般点积 ( W 是权重矩阵，通过 W 将查询向量投影到新的空间 ) 核方法 … … 用非线性函数将向量 q 和 k 映射到新空间 S_i = f(q, k) =

⎧⎩⎨⎪⎪qT\\sdotkiqT\\sdotki/d‾‾√qT\\sdotW\\sdotki核方法……………………点积缩放点积(d是键向量的维数)一般点积(W是权重矩阵，通过W将查询向量投影到新的空间)用非线性函数将向量q和k映射到新空间{qT\\sdotki……点积qT\\sdotki/d……缩放点积(d是键向量的维数)qT\\sdotW\\sdotki……一般点积(W是权重矩阵，通过W将查询向量投影到新的空间)核方法……用非线性函数将向量q和k映射到新空间

\\begin{cases} q^T \\sdot k\_i &\\dots\\dots &\\text{点积}\\\ q^T \\sdot k\_i /\\sqrt{d} &\\dots\\dots &\\text{缩放点积}(d\\text{是键向量的维数})\\\ q^T \\sdot W \\sdot k_i &\\dots\\dots &\\text{一般点积}(W\\text{是权重矩阵，通过}W\\text{将查询向量投影到新的空间})\\\ \\text{核方法} &\\dots\\dots &\\text{用非线性函数将向量}q\\text{和}k\\text{映射到新空间}\\\ \\end{cases} Si​=f(q,k)=⎩  ⎨  ⎧​qT⋅ki​qT⋅ki​/d ​qT⋅W⋅ki​核方法​……………………​点积缩放点积(d是键向量的维数)一般点积(W是权重矩阵，通过W将查询向量投影到新的空间)用非线性函数将向量q和k映射到新空间​  
相似度可以是查询和键的简单点积，也可以是缩放点积，其中 q q q 和 k k k 的点积除以每个键的维数 d d d 的平方根。这是最常用的两种计算相似度的技术。有时也经常使用权重矩阵 W W W 将查询投影到新空间，然后与键 k k k 进行点积。而核方法可以将非线性函数用作相似度计算。

**第二步**

第二步是找到权重 a a a。一般使用`SoftMax`完成。公式如下所示：  
a i = exp ⁡ ( S i ) ∑ j exp ⁡ ( S j ) (8) a\_i = \\frac{\\exp(S\_i)}{\\sum\_j\\exp(S\_j)}\\tag{8} ai​=∑j​exp(Sj​)exp(Si​)​(8)  
这里相似度连接到权重，就像一个全连接层。

**第三步**

第三步是 softmax ( a ) \\text{softmax} (a) softmax(a) 的结果与相应值 V V V 的加权组合。 a a a 的第一个值乘以 V V V 的第一个值，然后与 a a a 的第二个值与 V V V 的第二个值的乘积相加，依此类推。 最终输出结果就是我们所需的注意力值。  
attension value = ∑ i a i V i (9) \\text{attension value} = \\sum\_i a\_iV_i \\tag{9} attension value=i∑​ai​Vi​(9)  
**总结**

总结一下这 3 个步骤，在查询 q q q 和键 k k k 的帮助下，我们获得了注意力值，它是值 V V V 的加权和/线性组合，权重来自查询和键之间的某种相似性。

为了方便演示，上面的讲解我拿具体的向量值来演示计算过程。实际上，如果写成向量运算的形式，公式会更加简洁。  
Attension ( Q , K , V ) = softmax ( Q K T ) V (10) \\text{Attension}(Q, K, V) = \\text{softmax}(QK^T)V\\tag{10} Attension(Q,K,V)=softmax(QKT)V(10)  
在原始论文中，研究人员将自注意力矩阵除以 Q Q Q（或 K , V K,V K,V）维度的平方根，以防止内积变得过大。  
Attension ( Q , K , V ) = softmax ( Q K T d ) V (11) \\text{Attension}(Q, K, V) = \\text{softmax}\\Big(\\frac{QK^T}{\\sqrt{d}}\\Big)V \\tag{11} Attension(Q,K,V)=softmax(d ​QKT​)V(11)

#### 注意力机制的神经网络表示

![在这里插入图片描述](https://img-blog.csdnimg.cn/f5a5860a6df04b82b07744cf2aa13243.png#pic_center)

图4\. 注意力模块的神经网络表示

上图展示了注意力模块的神经网络表示。词嵌入首先被传递到线性层中，这些线性层没有“偏差”项，因此做的只是矩阵乘法。其中一层表示“键”，另一层表示“查询”，最后一层表示“值”。 在键和查询之间执行矩阵乘法，然后进行归一化，我们就得到了权重。接着将这些权重乘以值并相加，得到最终的注意力向量。这个模块可以在神经网络中使用，被称为“注意力块”。可以添加多个这样的注意力块以提供更多上下文。注意力块最大的优势是，我们可以获得梯度反向传播来更新注意力块（键、查询、值的权重）。

#### 掩码注意力

在机器翻译或文本生成任务中，我们经常需要预测下一个单词出现的概率，这类任务我们一次只能看到一个单词。此时注意力只能放在下一个词上，不能放在第二个词或后面的词上。简而言之，注意力不能有非平凡的超对角线分量。

我们可以通过添加掩码矩阵来修正注意力，以消除神经网络对未来的了解。  
Attension ( Q , K , V ) = softmax ( Q K T d + M ) V (12) \\text{Attension}(Q, K, V) = \\text{softmax}\\Big(\\frac{QK^T}{\\sqrt{d}}+M\\Big)V \\tag{12} Attension(Q,K,V)=softmax(d ​QKT​+M)V(12)  
其中 M M M 为掩码矩阵，其定义为：  
M = ( m i , j ) i , j = 0 n m i , j = { 0 i ≥ j − ∞ i < j (13) M = (m_{i,j})^n_{i,j=0}\\\ m_{i,j} =

{0−\\infini≥ji<j{0i≥j−\\infini<j

\\begin{cases} 0 & i \\ge j \\\ -\\infin & i \\lt j \\end{cases}\\tag{13} M=(mi,j​)i,j=0n​mi,j​={0−∞​i≥ji<j​(13)  
矩阵 M M M 的超对角线设置为负无穷大，以便 softmax 将其计算为 0。

#### 多头注意力

为了克服使用单一注意力的一些缺陷，研究人员又引入了多头注意力。让我们回到最开始的例子——_“小美长得很漂亮而且人还很好”_ 。这里“人”这个词，在语法上与“小美”和“好”这些词存在某种意义或关联。这句话中“人”这个词需要理解为“人品”，说的是小美的人品很好。仅仅使用一个注意力机制可能无法正确识别这三个词之间的关联，这种情况下，使用多个注意力可以更好地表示与“人”相关的词。这减少了注意力寻找所有重要词的负担，增加找到更多相关词的机会。

为此，让我们添加更多线性层作为键、查询和值。这些线性层并行训练，并且彼此具有独立的权重。图下图所示，每个值、键和查询都为我们提供了 3 个输出，而不是一个输出。这 3 组键和查询给出3种不同的权重。然后将这 3 个权重与 3 个值进行矩阵乘法，得到 3 个输出。 将这 3 个注意力连接起来，最终给出一个最终注意力输出。

![在这里插入图片描述](https://img-blog.csdnimg.cn/4e42e94dd5644718b36d5e85374a27f6.png#pic_center)

图5\. 具有 3 个线性层的多头注意力

上面演示中的 3 不是个定值，仅仅是为了演示选择的一个随机数。在实际场景中，这个值可以是任意数量的线性层，每一层被成为一个"头" ( h ) (h) (h)。也就是说，可以有任意数量 h h h 个线性层，提供 h h h 个注意力输出，然后将它们连接在一起。而这正是多头注意力（multiple heads）名称的由来。 下图是多头注意力的简化版，具有 h h h 头。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e7b455c57c734254b82c2840f2ee965e.png#pic_center)

图6\. 具有 h 层的多头注意力

理解了多头注意力的工作原理，那么多头注意力的公式表达就很简单，基本就是上图的结构：  
head i = Attension ( Q W i Q , K W i K , V W i V ) MultiHead = Concat ( head 1 , head 2 , … , head k ) W O (14) \\text{head}\_i = \\text{Attension}(QW\_i^Q, KW\_i^K, VW\_i^V)\\\ \\text{MultiHead} = \\text{Concat}(\\text{head}\_1, \\text{head}\_2, \\dots, \\text{head}_k)W^O \\tag{14} headi​=Attension(QWiQ​,KWiK​,VWiV​)MultiHead=Concat(head1​,head2​,…,headk​)WO(14)  
至此，我们已经介绍了查询、键、值、注意力和多头注意力背后的机制和思想，这些已经涵盖了 Transformer 网络的所有重要模块。在接下来，我们可以开始学习如何将这些模块组合在一起形成 Transformer 网络。

### Transformer 网络

Transformer 来自 Google 2017年发表的 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) 这篇论文。一经推出就收到业界极大关注。目前 Transformer 已经取代 RNN 成为 NLP 乃至计算机视觉（Vision Transformers）领域的最佳模型，当下炙手可热的 ChatGPT 就是从 Transformer 发展而来。

下图展示了 Transformer 的网络结构：

![在这里插入图片描述](https://img-blog.csdnimg.cn/ffce18497af74c1eb59dece6fcd82efd.png#pic_center)

图7\. Transformer Network

Transformer 网络由两部分组成——编码器和解码器。

在 NLP 任务中，编码器用于对初始句子进行编码，而解码器用于生成处理后的句子。Transformer 的编码器可以并行处理整个句子，因此比 RNN 更快更好——RNN 一次只能处理句子中的一个词。

#### 编码器

![在这里插入图片描述](https://img-blog.csdnimg.cn/dc827ebe86d74fbd8c020cdcd5dfd441.png#pic_center)

图8\. Transformer 网络的编码器部分

编码器网络从输入开始。 首先整个句子被一次性输入网络，然后将它们嵌入到“**输入嵌入**”块中。接着将“**位置编码**”添加到句子中的每个词。位置编码对理解句子中每个单词的位置至关重要。如果没有位置嵌入，模型会将整个句子视为一个装满词汇的袋子，没有任何顺序或意义。

##### 输入嵌入

句子中的每个词需要使用 embedding 空间来获得向量嵌入。嵌入只是将任何语言中的单词转换为其向量表示。举个例子，如图9所示，在 embedding 空间中，相似的词有相似的 embeddings，例如“猫”这个词和“喵”这个词在 embedding 空间中会落得很近，而“猫”和“芯片”在空间中会落得更远。

![在这里插入图片描述](https://img-blog.csdnimg.cn/615250c98ee04e9b9abc46c739a029a5.png#pic_center)

图9\. 输入嵌入

##### 位置编码

同一个词在不同的句子中可以表示不同的含义。 例如 _“你人真好”_，这句话中“人”这个词（位置 2）表示**人品**；而另语句 _“你是个好人”_ ，这句话中“人”这个词 （位置 5）表示**人类**。这两句话文字基本相同，但含义完全不同。为了帮助更好地理解语义，研究人员引入了位置编码。位置编码是一个向量，可以根据单词在句子中的上下文和位置提供信息。

在任何句子中，单词一个接一个地出现都蕴含着重要意义。如果句子中的单词乱七八糟，那么这句话很可能没有意义。但是当 Transformer 加载句子时，它不会按顺序加载，而是并行加载。由于 Transformer 架构在并行加载时不包括单词的顺序，因此我们必须明确定义单词在句子中的位置。这有助于 Transformer 理解句子词与词之间的位置。这就是位置嵌入派上用场的地方。位置嵌入是一种定义单词位置的向量编码。在进入注意力网络之前，将此位置嵌入添加到输入嵌入中。 图 10 给出了输入嵌入和位置嵌入在输入注意力网络之前的直观理解。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2099d919b0974310a501917758e01281.png#pic_center)

图10\. 位置嵌入的直观理解

有多种方法可以定义位置嵌入。在原始论文 [Attention is All You Need](https://arxiv.org/abs/1706.03762) 中，作者使用交替正余弦函数来定义位置嵌入，如图 5 所示。  
P E ( p o s , 2 i ) = sin ⁡ ( p o s 1000 0 2 i / d m o d e l ) P E ( p o s , 2 i + 1 ) = cos ⁡ ( p o s 1000 0 2 i / d m o d e l ) (15)

PE(pos,2i)PE(pos,2i+1)=sin(pos100002i/dmodel)=cos(pos100002i/dmodel)PE(pos,2i)=sin⁡(pos100002i/dmodel)PE(pos,2i+1)=cos⁡(pos100002i/dmodel)

\\begin{aligned} PE_{(pos, 2i)} &= \\sin(\\frac{pos}{10000^{2i/d_{model}}})\\\ PE_{(pos, 2i+1)} &= \\cos(\\frac{pos}{10000^{2i/d_{model}}}) \\end{aligned}\\tag{15} PE(pos,2i)​PE(pos,2i+1)​​=sin(100002i/dmodel​pos​)=cos(100002i/dmodel​pos​)​(15)  
其中 p o s pos pos 是位置， i i i 是维度。

这个嵌入算法在文本数据上效果很好，但它不适用于图像数据。因此可以有多种嵌入对象位置（文本/图像）的方法，并且可以在训练期间固定或学习它们。基本思想是，位置嵌入允许 Transformer 架构理解单词在句子中的位置，而不是通过混淆单词来混淆含义。

当输入嵌入和位置嵌入完成后，嵌入会流入编码器最重要的部分，其中包含两个重要的块——“多头注意力”和“前馈网络“。

##### 多头注意力

[多头注意力](#mha)的原理前面已经详细解释过，不清楚的可以点击 [这里](#mha) 学习回顾。

多头注意力块接收包含子向量（句子中的单词）的向量（句子）作为输入，然后计算每个位置与向量的所有其他位置之间的注意力。

![在这里插入图片描述](https://img-blog.csdnimg.cn/1e2b92f7658f4ecf8394274c06f7c4a4.png#pic_center)

图11\. 缩放点积注意力

上图展示了缩放点积注意力。缩放点击注意力跟自注意力非常相似，只是在第一次矩阵乘法（matmul）后加入了缩放（Scale）和掩码（Mask）。 原著论文中是这样定义缩放的：  
Scale = 1 / d output = Q T K / d (16) \\text{Scale} = 1/ \\sqrt d\\\ \\text{output} = Q^TK/ \\sqrt d \\tag{16} Scale=1/d ​output=QTK/d ​(16)  
其中 Q T K Q^TK QTK 是查询和键矩阵相乘后的结果， d d d 是词嵌入的维数。

缩放后的结果会传入掩码层。掩码层是可选的，对文本生成、机器翻译等任务很有用。

注意力模块的网络结构前面已经讲过，大家可以参考 [注意力机制的神经网络表示](#a)，这里不再赘述。

多头注意力接受多个键、查询和值，通过多个缩放点积注意力块提供多个注意力输出，最后连接多个注意力得到一个最终注意力输出。多头注意力前面也有详细解释，大家可以参考 [多头注意力](#mha)。

简单来说：主向量（句子）包含子向量（单词）——每个单词都有一个位置嵌入。注意力计算将每个单词视为一个“查询”，并找到与句子中其他单词相对应的“键”，然后对相应的“值”进行凸组合。在多头注意力中，选择多个值、查询和键，提供多重注意力（更好的词嵌入与上下文）。这些多重注意力被连接起来以给出最终的注意力值（所有多重注意力的所有单词的上下文组合），这比使用单个注意力块效果更好。

##### Add & Norm 与前馈

接下来的模块是 **Add & Norm**，它接收原始词嵌入的残差连接，将其添加到多头注意力的嵌入中，然后将其归一化为均值为0方差为 1的标准正态分布。

Add & Norm 的结果会送到 **前馈** 模块中，前馈模块后会再加一个 Add & Norm 块。

整个多头注意力和前馈模块在编码器中会重复 n n n 次（超参数）。

#### 解码器

![在这里插入图片描述](https://img-blog.csdnimg.cn/4214c07064904f69ab514935d7ca308f.png#pic_center)

图12\. Transformer 网络的解码器部分

编码器的输出也是一系列嵌入，且每个位置一个嵌入，其中每个位置嵌入不仅包含原始单词在该位置的嵌入，还包含它使用注意力学习到的其他单词的信息。

编码器的输出会发送到 Transformer 网络的解码器部分，如图 12 所示。解码器的目的是产生输出。在原作论文 [Attention is All You Need](https://arxiv.org/abs/1706.03762) 中，解码器被用于句子翻译（比如从汉语到英语）。所以编码器会接受中文句子，解码器会把它翻译成英文。在其他应用中，Transformer 网络的解码器部分不是必需的，因此我不会过多地阐述它。

Transformer 解码器按如下步骤工作（以原作论文中机器翻译任务为例）：

1.  在机器翻译任务中，解码器接受中文句子（用于中文到英文的翻译）。与编码器一样，首先需要添加一个词嵌入和一个位置嵌入并将其提供给多头注意力块。
2.  自注意力模块将为英文句子中的每个单词生成一个注意力向量，用于表示句子中一个单词与另一个单词的相关程度。
3.  然后将英文句子中的注意力向量与中文句子中的注意力向量进行比较。这是中文到英文单词映射发生的部分。
4.  在最后几层中，解码器预测将中文单词翻译成最可能的英文单词。
5.  整个过程重复多次以获得整个文本数据的翻译。

以上每一步与解码器网络模块的对应关系如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/b2f49b66af724b459230af75eaa527ad.png#pic_center)

图13\. 不同解码器块在句子翻译中的作用

解码器中大部分模块之前在编码器中都见过，这里不做过多的赘述。

### 用PyTorch实现Transformer

要从头构建我们自己的 Transformer 模型，需要遵循以下步骤：

1.  导入必要的库和模块
2.  定义基本模块：多头注意力、位置前馈网络、位置编码
3.  构建编码器层和解码器层
4.  将编码器层和解码器层合在一起构建完整的 Transformer 模型
5.  准备样本数据
6.  训练模型

我们一步一步来完成上面的工作。

#### 导入必要的库和模块

让我们从导入必要的库和模块开始。构建 Transformer 需要用到如下库和模块：

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    import math
    import copy
    

#### 定义基础模块

接着，我们将定义 Transformer 模型的基础模块。

##### 多头注意力

多头注意力前面已经详细讲过，其结构参见图6。简单来说，多头注意力机制计算序列中每对位置之间的注意力。它由多个“注意力头”组成，捕捉输入序列的不同方面。

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super(MultiHeadAttention, self).__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_probs, V)
            return output
            
        def split_heads(self, x):
            batch_size, seq_length, d_model = x.size()
            return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
            
        def combine_heads(self, x):
            batch_size, _, seq_length, d_k = x.size()
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
            
        def forward(self, Q, K, V, mask=None):
            Q = self.split_heads(self.W_q(Q))
            K = self.split_heads(self.W_k(K))
            V = self.split_heads(self.W_v(V))
            
            attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
            output = self.W_o(self.combine_heads(attn_output))
            return output
    

`MultiHeadAttention` 类使用输入参数和线性变换层初始化模块。它计算注意力分数，将输入张量重塑为多个头，并组合所有头的注意力输出。`forward()` 方法计算多头自注意力，允许模型关注输入序列的一些不同方面。

##### 位置前馈网络

    class PositionWiseFeedForward(nn.Module):
        def __init__(self, d_model, d_ff):
            super(PositionWiseFeedForward, self).__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))
    

`PositionWiseFeedForward` 类扩展了 PyTorch 的 `nn.Module` 并实现了位置前馈网络。该类使用两个线性变换层和一个 ReLU 激活函数进行初始化。`forward()` 方法按顺序应用这些转换和激活函数来计算输出。此过程使模型能够在进行预测时考虑输入元素的位置。

##### 位置编码

位置编码用于注入每个token在输入序列中的位置信息。它使用不同频率的正弦和余弦函数来生成位置编码。

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_seq_length):
            super(PositionalEncoding, self).__init__()
            
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))
            
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]
    

`PositionalEncoding` 类使用输入参数 `d_model` 和 `max_seq_length` 进行初始化，创建一个张量来存储位置编码值。该类根据比例因子 `div_term` 分别计算偶数和奇数索引的正弦值和余弦值。`forward()` 方法通过将存储的位置编码值添加到输入张量来计算位置编码，从而使模型能够捕获输入序列的位置信息。

有了这些基础模块，我们就可以开始构建编码器层和解码器层了。

#### 编码器层

Tranformer 的编码器层结构参见 [图8\. Transformer网络的编码器部分](#encoder)。编码器层由一个多头注意层、一个位置前馈层和两个层归一化层组成。

    class EncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout):
            super(EncoderLayer, self).__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, mask):
            attn_output = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            return x
    

`EncoderLayer` 类使用输入参数和组件进行初始化，包括多头注意力模块、位置前馈网络模块、两层归一化模块和 dropout 层。`forward()` 方法通过应用自注意力、将注意力输出添加到输入张量并对结果进行归一化来计算编码器层输出。然后，它计算位置前馈输出，将其与归一化的自注意力输出相结合，并在返回处理后的张量之前对最终结果进行归一化。

#### 解码器层

Tranformer 的解码器层结构参见 [图9\. Transformer网络的解码器部分](#decoder)。解码器层由两个多头注意层、一个位置前馈层和三个层归一化层组成。

    class DecoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout):
            super(DecoderLayer, self).__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            self.cross_attn = MultiHeadAttention(d_model, num_heads)
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, enc_output, src_mask, tgt_mask):
            attn_output = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout(attn_output))
            attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
            x = self.norm2(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout(ff_output))
            return x
    

`DecoderLayer` 类使用输入参数和组件进行初始化，例如用于屏蔽自注意力和交叉注意力的多头注意力模块、位置前馈网络模块、三层归一化模块和 dropout 层。

`forward()` 方法通过执行以下步骤计算解码器层输出：

1.  计算掩码自注意力输出并将其添加到输入张量中，然后进行dropout和layer normalization。
2.  计算解码器和编码器输出之间的交叉注意力输出，并将其添加到归一化的掩码自注意力输出，然后进行 dropout 和层归一化。
3.  计算位置前馈输出并将其与归一化的交叉注意力输出相结合，然后进行 dropout 和层归一化。
4.  返回处理后的张量。

这些操作使解码器能够根据输入和编码器输出生成目标序列。

#### Transformer模型

有了编码器和解码器后，我们就可以将编码器和解码器结合起来创建完整的 Transformer 模型。

    class Transformer(nn.Module):
        def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
            super(Transformer, self).__init__()
            self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
            self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
    
            self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
            self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    
            self.fc = nn.Linear(d_model, tgt_vocab_size)
            self.dropout = nn.Dropout(dropout)
    
        def generate_mask(self, src, tgt):
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
            seq_length = tgt.size(1)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
            tgt_mask = tgt_mask & nopeak_mask
            return src_mask, tgt_mask
    
        def forward(self, src, tgt):
            src_mask, tgt_mask = self.generate_mask(src, tgt)
            src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
    
            enc_output = src_embedded
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
    
            dec_output = tgt_embedded
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
    
            output = self.fc(dec_output)
            return output
    

`Transformer` 类将前面定义的模块组合在一起，创建一个完整的 Transformer 模型。在初始化期间，Transformer 模块设置输入参数并初始化各种组件，包括源序列和目标序列的嵌入层、位置编码模块、用于创建堆叠层的编码层和解码层模块、用于投影解码器输出的线性层和 dropout 层。

`generate_mask()` 方法为源序列和目标序列创建二进制掩码，用于忽略填充标记并防止解码器处理未来的标记。 `forward()` 方法通过以下步骤计算 Transformer 模型的输出：

1.  使用 `generate_mask()` 方法生成源和目标掩码。
2.  计算源和目标嵌入，并应用位置编码和 dropout。
3.  通过编码器层处理源序列，更新 `enc_output` 张量。
4.  通过解码器层处理目标序列，使用 `enc_output` 和掩码，并更新 `dec_output` 张量。
5.  将线性投影层应用于解码器输出，获得最终输出。

以上步骤使 Transformer 模型能够处理输入序列并根据其组件的组合功能生成输出序列。

#### 准备样本数据

    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1
    
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    
    # 生成随机样本数据
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  
    

为了方便演示，我这里随机生成样本数据。在实际开发中，您可以使用更大的数据集，预处理文本。

#### 训练模型

准备好数据后就可以训练模型了。

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    transformer.train()
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    

以上就是如何使用 Pytorch 中从头开始构建一个简单的 Transformer。

### 总结

所有大型语言模型都使用 Transformer 编码器或解码器块进行训练。 因此，了解深入理解 Transformer 网络非常重要。希望本文对您有所帮助。


参考：[深度解析 Transformer 和注意力机制](https://blog.csdn.net/jarodyv/article/details/130867562)
