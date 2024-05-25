图像融合旨在通过将高分辨率图像与有限的光谱信息和具有丰富光谱数据的低分辨率图像相结合来生成高分辨率多/高光谱图像。目前基于深度学习的图像融合方法主要依靠cnn或transformer提取特征并合并不同类型的数据。虽然 CNN 是有效的，但它们的感受野是有限的，限制了它们捕获全局上下文的能力。相反，变形金刚擅长学习全局信息，但受到二次复杂性的阻碍。幸运的是，状态空间模型 (SSM) 的最新进展，特别是 Mamba，通过启用线性复杂度的全局意识，为这个问题提供了一个有前途的解决方案。然而，很少有人尝试探索SSM在信息融合方面的潜力，这是图像融合等领域的关键能力。因此，我们提出了FusionMamba，一种用于高效图像融合的创新方法。我们的贡献主要集中在两个方面。首先，认识到来自不同来源的图像具有不同的属性，我们将 Mamba 块合并到两个 U 形网络中，提出了一种新颖的架构，该架构以高效、独立和分层的方式提取空间和光谱特征。其次，为了有效地结合空间和光谱信息，我们扩展了 Mamba 块以适应双重输入。这种扩展导致创建了一个名为 FusionMamba 块的新模块，其性能优于现有的融合技术，例如连接和交叉注意力。为了验证 FusionMamba 的有效性，我们对与三个图像融合任务相关的五个数据集进行了一系列实验。定量和定性评估结果表明，我们的方法达到了最先进的性能（SOTA）性能，强调了 FusionMamba 的优越性。