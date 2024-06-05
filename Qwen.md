# Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond

<img width="906" alt="image" src="https://github.com/icey-zhang/notebook/assets/54712081/887eea64-ca5e-423c-a969-4754ae984d71">


[[Paper]](https://arxiv.org/abs/2308.12966) [[Code]](https://github.com/QwenLM/Qwen/tree/main) [[Document]](https://qwen.readthedocs.io/zh-cn/latest/)

### 论文总结

这篇论文介绍了Qwen-VL系列，这是一组基于Qwen-7B语言模型的高性能和多功能视觉语言基础模型。该系列包括视觉接收器、语言对齐视觉编码器和位置感知适配器，并通过精心设计的三阶段训练管道对大量图文语料库进行优化。这些模型能够执行各种视觉语言任务，如图像描述、问题回答、面向文本的问题回答和视觉定位。

### 方法部分

#### 模型架构

Qwen-VL的整体网络架构包括三个组件，详细参数如下：

1. **大语言模型（LLM）**：
   Qwen-VL采用Qwen-7B大语言模型作为基础组件，模型初始化为预训练的Qwen-7B权重。

2. **视觉编码

### 论文总结

这篇论文介绍了Qwen-VL系列，这是一组基于Qwen-7B语言模型的高性能和多功能视觉语言基础模型。该系列模型通过引入视觉接收器、语言对齐视觉编码器和位置感知适配器，使大语言模型具备视觉能力。通过精心设计的三阶段训练管道，Qwen-VL能够执行多种视觉语言任务，如图像描述、问题回答、文本导向问题回答和视觉定位。

### 方法部分

#### 模型架构

Qwen-VL的整体网络架构包括三个主要组件：

1. **大语言模型（LLM）**：
   - 使用Qwen-7B作为基础模型，初始化时使用预训练的Qwen-7B权重。
   - 模型参数详情见表1：
     - 视觉编码器：1.9B参数
     - VL适配器：0.08B参数
     - LLM：7.7B参数
     - 总计：9.6B参数

2. **视觉编码器**：
   - 使用Vision Transformer（ViT）架构，初始化时使用Openclip的ViT-bigG预训练权重。
   - 输入图像被调整为特定分辨率，处理后分割为补丁，每个补丁生成一组图像特征。

3. **位置感知视觉语言适配器**：
   - 引入单层交叉注意模块，随机初始化，通过一组可训练向量作为查询向量，与视觉编码器生成的图像特征进行交叉注意操作。
   - 压缩视觉特征序列至固定长度（256），并在交叉注意机制的查询-键对中加入二维绝对位置编码，以减轻压缩过程中可能丢失的位置细节。

#### 输入输出

- **图像输入**：通过视觉编码器和适配器处理，生成固定长度的图像特征序列。在图像特征序列的开头和结尾分别添加特殊标记（<img> 和 </img>），表示图像内容的开始和结束。
- **边界框输入和输出**：用于细粒度视觉理解和定位，Qwen-VL训练过程中涉及区域描述、问题和检测数据。将边界框规范化处理为字符串格式，并添加特殊标记（<box> 和 </box>）以区分检测字符串和常规文本。

### 训练过程

Qwen-VL模型的训练分为三个阶段：两阶段的预训练和最终的指令微调训练。

1. **第一阶段预训练**：
   - 使用大规模、弱标记的网络抓取图文对数据集。原始数据集包含50亿图文对，清理后剩余1.4亿数据，其中77.3%为英文，22.7%为中文。

2. **多任务预训练**：
   - 引入高质量和细粒度的视觉语言注释数据，输入分辨率更高，数据为图文交错形式。训练任务包括图像描述、视觉问答、定位等，共七个任务。

3. **监督微调**：
   - 微调阶段通过指令微调增强模型的指令跟随和对话能力，最终生成Qwen-VL-Chat模型。多模态指令微调数据主要来自于通过LLM自指令生成的描述数据或对话数据。

### 实验结果

1. **图像描述和常规视觉问答**：
   - 在多个基准测试上，Qwen-VL和Qwen-VL-Chat在图像描述和视觉问答任务中表现出色。

2. **文本导向视觉问答**：
   - 在多个基准上，Qwen-VL系列模型在文本导向视觉理解任务中显示了较高的性能。

3. **参考表达理解**：
   - Qwen-VL系列模型在细粒度图像理解和定位任务中表现优异，在多个基准测试中取得顶级结果。

这篇论文通过引入新的模型架构和训练方法，有效提升了视觉语言模型的性能，使其在多个任务上取得了领先的表现。

【注】请在引用或进一步使用这些内容时确保对相关数据和方法进行详细核对，以保证准确性和完整性。