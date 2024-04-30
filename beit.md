BEiT：视觉BERT预训练模型


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


1 BERT 方法回顾

