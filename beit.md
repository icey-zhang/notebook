BEiT：视觉BERT预训练模型


本文目录
> 1 BERT 方法回顾

> 2 BERT 可以直接用在视觉任务上吗？

> 3 BEiT 原理分析
>>  3.1 将图片表示为 image patches
>>  3.2 将图片表示为 visual tokens
>>>  3.2.1 变分自编码器 VAE
>>>   3.2.2 BEIT 里的 VAE：tokenizer 和 decoder
>>>  3.2.3 BEIT 的 Backbone：Image Transformer
>>> 3.2.4 类似 BERT 的自监督训练方式：Masked Image Modeling
>>> 3.2.5 BEIT 的目标函数：VAE 视角
>>> 3.2.6 BEIT 的架构细节和训练细节超参数
>>> 3.2.7 BEIT 在下游任务 Fine-tuning
>>> 3.2.8 实验
