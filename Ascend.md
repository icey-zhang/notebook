![image](https://github.com/user-attachments/assets/e9a0b24f-36f8-4380-9039-76963bd84810)## 昇腾AI全栈软硬件平台
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
![image](https://github.com/user-attachments/assets/51ccf092-546c-496b-9161-9f2cbe239358)

- MindX SDK： **帮助特定领域的用户快速开发并部署人工智能应用**，比如工业质检、检索聚类等，致力于简化昇腾 AI 处理器推理业务开发过程，降低使用昇腾AI处理器开发的门槛。

![image](https://github.com/user-attachments/assets/49cc2369-968e-482a-b26e-ed702ea96f06)


- MindX DL（昇腾深度学习组件）： 是支持 Atlas训练卡、推理卡的深度学习组件，提供昇腾 AI 处理器集群调度、昇腾 AI 处理器性能测试、模型保护等基础功能，快速使能合作伙伴进行深度学习平台开发。

![image](https://github.com/user-attachments/assets/d8ee6ee5-d023-4186-90ea-847fb535e347)

- MindX Edge（昇腾智能边缘组件）： 提供边缘 AI 业务容器的全生命周期管理能力，同时提供严格的安全可信保障，为客户提供边云协同的边缘计算解决方案，使能客户快速构建边缘 AI 业务。

![image](https://github.com/user-attachments/assets/77539b19-1d90-44e2-9e6f-12bc355280ff)

- Modelarts： ModelArts 是面向开发者的一站式 AI 平台，为机器学习与深度学习提供海量数据预处理及交互式智能标注、大规模分布式训练、自动化模型生成，及端-边-云模型按需部署能力，帮助用户快速创建和部署模型，管理全周期 AI 工作流。

![image](https://github.com/user-attachments/assets/3db27caa-1f76-48e8-ac92-72fa4d93b208)

- HiAI Service：HUAWEI HiAI是面向智能终端的AI能力开放平台，基于 “芯、端、云”三层开放架构，即芯片能力开放、应用能力开放、服务能力开放，构筑全面开放的智慧生态，让开发者能够快速地利用华为强大的AI处理能力，为用户提供更好的智慧应用体验。

  行业应用
主要应用于能源、金融、交通、电信、制造、医疗等行业，这里就不过多介绍了。

安装 MindSpore 和 MindFormers 简单流程
建议：确定要安装的MindSpore具体版本，再确定需要安装的驱动和固件版本。
主要有物理机、容器和虚拟机安装。其中，容器和虚拟机不支持固件包安装。

### 参考
[GPT2 模型推理](https://zhuanlan.zhihu.com/p/637918406)
