[【参考】](https://blog.csdn.net/weixin_44211968/article/details/121314757)
## 元学习

了解元学习的详情，请参考我的另一篇文章：[【机器学习】Meta-Learning（元学习）](https://blog.csdn.net/weixin_44211968/article/details/121313918?spm=1001.2014.3001.5501)

Meta learning （元学习）中，在 meta training 阶段将数据集分解为不同的 meta task，去学习类别变化的情况下模型的泛化能力，在 meta testing 阶段，面对全新的类别，不需要变动已有的模型，就可以完成分类。

形式化来说，few-shot 的训练集中包含了很多的类别，每个类别中有多个样本。在训练阶段，会在训练集中随机抽取 C 个类别，每个类别 K 个样本（总共 CK 个数据），构建一个 meta-task，作为模型的支撑集（support set）输入；再从这 C 个类中剩余的数据中抽取一批（batch）样本作为模型的预测对象（batch set）。即要求模型从 C*K 个数据中学会如何区分这 C 个类别，这样的任务被称为 C-way K-shot 问题。

训练过程中，每次训练（episode）都会采样得到不同 meta-task，所以总体来看，训练包含了不同的类别组合，这种机制使得模型学会不同 meta-task 中的共性部分，比如如何提取重要特征及比较样本相似等，忘掉 meta-task 中 task 相关部分。通过这种学习机制学到的模型，在面对新的未见过的 meta-task 时，也能较好地进行分类。
