[【参考】](https://blog.csdn.net/weixin_44211968/article/details/121314757)
## 元学习

了解元学习的详情，请参考我的另一篇文章：[【机器学习】Meta-Learning（元学习）](https://blog.csdn.net/weixin_44211968/article/details/121313918?spm=1001.2014.3001.5501)

Meta learning （元学习）中，在 meta training 阶段将数据集分解为不同的 meta task，去学习类别变化的情况下模型的泛化能力，在 meta testing 阶段，面对全新的类别，不需要变动已有的模型，就可以完成分类。

形式化来说，few-shot 的训练集中包含了很多的类别，每个类别中有多个样本。在训练阶段，会在训练集中随机抽取 C 个类别，每个类别 K 个样本（总共 CK 个数据），构建一个 meta-task，作为模型的支撑集（support set）输入；再从这 C 个类中剩余的数据中抽取一批（batch）样本作为模型的预测对象（batch set）。即要求模型从 C*K 个数据中学会如何区分这 C 个类别，这样的任务被称为 C-way K-shot 问题。

训练过程中，每次训练（episode）都会采样得到不同 meta-task，所以总体来看，训练包含了不同的类别组合，这种机制使得模型学会不同 meta-task 中的共性部分，比如如何提取重要特征及比较样本相似等，忘掉 meta-task 中 task 相关部分。通过这种学习机制学到的模型，在面对新的未见过的 meta-task 时，也能较好地进行分类。

## 少样本学习中的相关概念
概念1：Support set VS training set
小样本带标签的数据集称为support set，由于support set数据样本很少，所以不足以训练一个神经网络。而training set每个类别样本量很大，使用training set训练的模型能够在测试集取得很好的泛化效果。

概念2：Supervised learning VS few-shot learning
监督学习：
（1）测试样本之前从没有见过
（2）测试样本类别出现在训练集中
Few-shot learning
（1）query样本之前从没有见过
（2）query样本来自于未知类别

由于query并未出现在训练集中，我们需要给query提供一个support set，通过对比query和support set间的相似度，来预测query属于哪一类别。

概念3：k-way n-shot support set
k-way：support set中有 k 个类别
n-shot：每一个类别有 n 个样本

Few-shot learning的预测准确率随 way 增加而减小，随 shot 增加而增加。因为对于2-way问题，预测准确率显然要比1000-way问题要高。而对于 shot，一个类别中样本数越多越容易帮助模型找到正确的类别。

少样本学习的基本思想
Few-shot learning的最基本的思想是学一个相似性函数： 来度量两个样本和的相似性。 越大表明两个图片越相似，越小，表明两个图片差距越大。

操作步骤：
（1）从大规模训练数据集中学习相似性函数
（2）比较query与support set中每个样本的相似度，然后找出相似度最高的样本作为预测类别。





