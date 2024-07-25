## 昇腾AI全栈软硬件平台
### 简述
昇腾芯片是华为公司发布的两款 AI 处理器(NPU)，**昇腾910（用于训练）和昇腾310（用于推理）处理器**，采用自家的**达芬奇架构**。昇腾在国际上对标的主要是英伟达的GPU，国内对标的包括**寒武纪、海光**等厂商生产的系列AI芯片产品（如：思元590、深算一号等）。

整个昇腾软硬件全栈包括5层，自底向上为**Atlas系列硬件、异构计算架构、AI框架、应用使能、行业应用。**

![image](https://github.com/user-attachments/assets/58c11500-75fc-4d3d-a964-570ef4c16856)

![image](https://github.com/user-attachments/assets/8f47451c-d1c9-4a78-90f4-2f8867f434db)

### Atlas系列硬件
Atlas系列产品是基于昇腾910和昇腾310打造出来的、面向不同应用场景（端、边、云）的系列AI硬件产品。比如：

Atlas 800（型号：9000） 是训练服务器，包含**8个训练卡（Atlas 300 T：采用昇腾910）**。
Atlas 900 是训练集群（由**128台Atlas 800（型号：9000）构成**），相当于是由一批训练服务器组合而成。
Atlas 800（型号：3000） 是推理服务器，包含**8个推理卡（Atlas 300 I：采用昇腾310）**。

## 异构计算架构
**异构计算架构（CANN）是对标英伟达的CUDA + CuDNN的核心软件层**，对上支持多种AI框架，对下服务AI处理器，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台，主要包括有各种引擎、编译器、执行器、算子库等。之所以叫异构软件，是因为承载计算的底层硬件包括AI芯片和通用芯片，自然就需要有一层软件来负责算子的调度、加速和执行，最后自动分配到对应的硬件上（CPU或NPU）。

![image](https://github.com/user-attachments/assets/c97f7bcb-a06f-4853-8fac-7f9a7c0d9a01)

- **昇腾计算语言**（Ascend Computing Language，AscendCL）接口是昇腾计算开放编程框架，对开发者屏蔽底层多种处理器差异，提供算子开发接口TBE、标准图开发接口AIR、应用开发接口，支持用户快速构建基于Ascend平台的AI应用和业务。
- **昇腾计算服务层**主要提供**昇腾算子库AOL**，通过神经网络（Neural Network，NN）库、线性代数计算库（Basic Linear Algebra Subprograms，BLAS）等高性能算子加速计算；昇腾调优引擎AOE，通过算子调优OPAT、子图调优SGAT、梯度调优GDAT、模型压缩AMCT提升模型端到端运行速度。同时提供AI框架适配器**Framework Adaptor**用于兼容Tensorflow、Pytorch等主流AI框架。
- **昇腾计算编译层**通过图编译器（Graph Compiler）将用户输入中间表达（Intermediate Representation，IR）的计算图编译成昇腾硬件可执行模型；同时借助张量加速引擎TBE（Tensor Boost Engine）的自动调度机制，高效编译算子。
- **昇腾计算执行层**负责模型和算子的执行，提供运行时库（Runtime）、图执行器（Graph Executor）、数字视觉预处理（Digital Vision Pre-Processing，DVPP）、人工智能预处理（Artificial Intelligence Pre-Processing，AIPP）、华为集合通信库（Huawei Collective Communication Library，HCCL）等功能单元。
- **昇腾计算基础层**主要为其上各层提供基础服务，如共享虚拟内存（Shared Virtual Memory，SVM）、设备虚拟化（Virtual Machine，VM）、主机-设备通信（Host Device Communication，HDC）等。


AI框架层主要包括**自研框架MindSpore（昇思）** 和第三方框架（PyTorch、TensorFlow等） ，其中MindSpore完全由华为自主研发，第三方框架华为只是做了适配和优化，让PyTorch和TensorFlow等框架编写的模型可以高效的跑在昇腾芯片上。

以PyTorch为例，华为的框架研发人员会将其做好适配，然后把适配后的PyTorch源码发布出来，想要在昇腾上用PyTorch的开发者，下载该源码自行编译安装即可。

应用使能
应用使能层主要包括ModelZoo、MindX SDK、MindX DL、MindX Edge等。

- ModelZoo： **存放模型的仓库**

- MindX SDK： **帮助特定领域的用户快速开发并部署人工智能应用**，比如工业质检、检索聚类等，致力于简化昇腾 AI 处理器推理业务开发过程，降低使用昇腾AI处理器开发的门槛。


- 
