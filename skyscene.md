# 摘要
先前关于**遥感基础模型** (RSFM) 的研究揭示了地球观测通用模型的巨大潜力。然而，**这些工作主要集中在没有时间和地理上下文建模的单一模态上，阻碍了它们对不同任务的能力。**在这项研究中，我们提出了 SkySense，这是一种通用的**十亿尺度模型**，在具有 **21.5 亿个时间序列的精选多模态遥感图像 (RSI) 数据集上进行预训练**。SkySense 结合了分解的多模态时空编码器，将光学和合成孔径雷达 (SAR) 数据的时间序列作为输入。该编码器由我们提出的多粒度对比学习预训练，以学习不同模态和空间粒度的表示。为了进一步通过地理上下文线索增强 RSI 表示，我们引入了 **Geo-Context Prototype Learning** 在 RSI 的多模态时空特征上学习**区域感知原型**。据我们所知，SkySense 是迄今为止最大的多模态 RSFM，其模块可以灵活地组合或单独使用以适应各种任务。它在彻底评估方面表现出卓越的泛化能力，包括 7 个任务的 16 个数据集，从单模态到多模态、静态到时间和分类到定位的分类。SkySense 在所有测试场景中都超过了最近的 18 个 RSFM。具体来说，它大大优于 GFM、SatLas 和 Scale-MAE 等最新模型，即平均分别为 2.76%、3.67% 和 3.61%。我们将发布预训练的权重以促进未来的研究和地球观测应用。

# 介绍
遥感图像(RSI)解释对于理解我们常见的家地球[17,70]至关重要，通过完全不同的任务 [5, 14, 49, 84]，例如作物监测、自然灾害管理等。每个任务可能需要大量专门的努力和资源来构建特定于任务的模型。最近，基础模型作为预训练的通用模型出现，该模型在广泛的下游任务中表现出色 [82, 88]。因此，人们对探索许多地球观测(EO)任务的综合遥感基础模型(RSFM)很感兴趣。
关键问题自然出现：RSFM 至关重要。首先，**理想的RSFM应该具备感知多模态时间RSI的能力**。EO严重依赖于遥感数据的多模态时间序列，包括时间光学和合成孔径雷达(SAR)数据。单独的模态提供了独特的优势和相互补充。例如，光学图像提供了丰富的光谱波段和纹理细节，但容易受到天气的影响[89]。但是，SAR传感器可以捕获所有天气条件下的真实清晰图像[34,42]。此外，此类数据的时间序列为各种任务 [5, 24, 85] 提供了关键的时间线索，例如变化检测。其次，在不同的空间(即像素、对象和图像级)粒度下，使用不同的模式(即单模态和多模态)部署EO任务时，RSFM应该很容易定制。最后但并非最不重要的一点是，遥感数据本质上取决于它们的时空坐标，这提供了丰富的区域和季节性地理背景，有利于RSI解释，如[12,27,35,43,44]所示。因此，RSFM应具有有效的地理上下文学习和利用的重要能力。以前关于 RSFM [1, 2, 4, 8, 19, 37, 51–54,57,64, 66, 72, 73, 75, 77] 的工作已经证明了它们在几个特定数据集上的初步成功。**然而，由于单模态预训练和忽略地理上下文等因素**，这些 RSFM 在 EO 任务中的应用受到限制。在本文中，我们提出了 SkySense，这是一种十亿规模的多模态遥感基础模型 (MMRSFM)。SkySense 包含 2206 亿个参数，并在大规模多模态数据集上进行了预训练，该数据集包含从高空间分辨率光学图像 (HSROI)、中分辨率时间多光谱图像 (TMSI) 和时间 SAR 图像 (TSARI) 中提取的 2150 万个 RSI 时间序列。为了处理**多模态时间RSI序列**，SkySense采用分解的多模态时空编码器独立进行空间特征提取和多模态时间融合，因为RSI序列本质上是空间对齐的。这导致了模块化设计，允许灵活使用其模块，**即空间编码器既可以单独使用，也可以结合融合模块来支持从静态单模态到时间多模态的任务**。这种设计提供了RSI序列的强建模，同时与普通的3D结构相比，使用的参数要少得多[50,87]。通过多粒度对比学习对分解后的编码器进行预训练，构建来自不同模态和空间粒度的特征。此外，我们提出了 **Geo-Context Prototype Learning** 从**给定地理位置的 RSI 特征中生成区域原型**。**这种方法通过利用隐藏在众多未标记 RSI 中的区域上下文线索来增强多模态时空表示学习**。SkySense 在各种模式和 EO 任务中取得了最先进的性能（SOTA），如图 1 所示。我们在一组不同的 16 数据集 [9, 15, 16, 19, 20, 24, 40, 62, 65, 68, 78, 80] 上评估 SkySense，其中选择涵盖不同的任务类型、模式和空间尺度。结果表明，SkySense 在所有测试场景中都优于 18 个高级 RSFM [1, 2, 4, 8, 19, 51–54,57,64, 66, 72, 73, 75, 77]，验证了其在广泛的 EO 解释任务中的竞争边缘。表格 1 将我们的工作与最新的代表性研究进行了比较，即各种输入类型的 EO 解释。
<img width="451" alt="image" src="https://github.com/icey-zhang/notebook/assets/54712081/cd337f25-e240-4ced-b4b5-ca8a7b08cb0e">

总之，我们的技术贡献是： 
• 我们提出了 SkySense，这是迄今为止最大的 MM-RSFM，具有模块化设计，能够处理不同的任务，从单模态到多模态、静态到时间和分类到定位。

• SkySense 的设计涉及三个新颖的技术组件：a) 分解的多模态时空编码器，以有效地处理多模态时间 RSI； b) 多粒度对比学习，它学习不同粒度级别的特征以促进不同的任务； c) Geo-Context Prototype Learning 提取区域感知地理上下文线索以实现隐式地理知识集成。

• 我们将 SkySense 与最近发布的 18 个 RSFM 进行了广泛的比较。我们的模型实现了 SOTA 性能，平均支持 GFM、SatLas 和 Scale-MAE 等最新模型超过 2.5%。我们希望预训练权重的释放将有助于遥感社区并促进未来的研究。

# 相关工作
最近的遥感基础模型从视觉基础模型的研究中获得灵感[3,7,11,21,26,28 -30,45,56,71]。遥感数据固有地集成了时空坐标，具有不同的空间尺度。主流 RSFM 将基础模型技术扩展到时空 RS 数据，例如对比学习。例如，GASSL[2]在MoCo-v2框架[13]中利用地理位置预测作为额外的前文本任务。DINO-MC[77]利用不同大小的多个视图进行DINO框架[7]中的自我监督学习。SeCo [52] 和 CACo [51] 都提出了对比学习，通过使用时间 RSI 序列的时空结构来感知短期和长期变化。此外，也有改进基于mimi的框架[57,64,72]或探索模型放大[8]的工作。例如，RingMo[64]修改了MAE以适应密集目标检测。SatMAE [19] 采用 TMsi 来提高时间序列的性能。Scale-MAE[57]构建了一个具有尺度感知编码器的框架。最近的努力，如 CMID [54] 和 GFM [53] 已经开始探索 CL 和 MIM 策略的融合。同时，CROMA [23] 和 DeCUR [74] 使用静态图像研究了单模态和多模态任务的多模态预训练。在这项研究中，我们提出了一个全面的 MM-RSFM、SkySense，来填补现有 RSFM 的差距，即 RingMo、CACo 的单一模态等，Scale-MAE、CROMA 等的静态输入，而忽略了 SatLas、RVSA 等的地理背景。