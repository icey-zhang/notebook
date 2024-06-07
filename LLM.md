# 多模态大模型
链接：https://deepshare.feishu.cn/wiki/DXGzwWmHyitwdLkTftdcFtbxnNd?from=from_copylink

链接：https://pan.baidu.com/s/1zOCAJvUwhz1_HDcDjOOrFA?pwd=6666 
提取码：6666 
--来自百度网盘超级会员V1的分享


# llM-面试八股-q&a

百度网盘分享链接：

链接: https://pan.baidu.com/s/1ymoF1Qj6KXxO80pnBAIK6A 

提取码: 5btt

# [LLM PTQ量化经典研究解析](https://mp.weixin.qq.com/s/01rDsMHY6pBHmGhwZhouvQ)
## GPTQ
- 论文: GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS
- 代码: https://github.com/IST-DASLab/gptq
- 类型: W4A16
- 描述: GPTQ采用海森矩阵计算量化误差，优化权重更新，使用Cholesky分解加速过程，有效解决了权重优化问题。
## LLM.int8()
- 论文: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- 代码: https://github.com/TimDettmers/bitsandbytes
- 类型: W8A8
- 描述: LLM.int8()通过离群点检测，将含有异常值的部分用fp16计算，剩余部分用int8量化，实现了近乎无损的精度。
## SmoothQuant
- 论文: SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
- 代码: https://github.com/mit-han-lab/smoothquant, https://github.com/Guangxuan-Xiao/torch-int
- 类型: W8A8
- 描述: SmoothQuant通过平滑weight和channel维度的activation，使得量化更加容易，成为W8A8的业界主流。
## AWQ
- 论文: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
- 代码: https://github.com/mit-han-lab/llm-awq
- 类型: W4A16
- 描述: AWQ通过激活值发现重要weight，对weight进行per-channel的scale，对activation除以scale，寻找最小化量化误差的scale，是当前W4A16的SOTA。
## ZeroQuant系列
- 代码: https://github.com/microsoft/DeepSpeed
- 描述: ZeroQuant系列由微软DeepSpeed团队推出，包括对weight和activation的不同量化方法，引入了低秩补偿(LoRC)技术，探索了FP8和FP4格式量化，以及硬件增强型量化框架。
## SpQR
- 论文: SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
- 代码: https://github.com/Vahe1994/SpQR
- 类型: W4A16
- 描述: SpQR通过计算参数敏感度，对敏感weight和异常值进行特殊处理，存储在更高精度中，其余权重压缩为3-4比特，形成一种压缩表示。
## OWQ
- 论文: OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models
- 代码: https://github.com/xvyaward/owq
- 类型: W4A16
- 描述: OWQ使用海森矩阵计算敏感度，将敏感列用fp16存储，其余低比特量化，提出对特定任务finetune采取对弱列微调的方式，节省内存开销。
## SqueezeLLM
- 论文: SqueezeLLM: Dense-and-Sparse Quantization
- 代码: https://github.com/SqueezeAILab/SqueezeLLM
- 类型: W4A16
- 描述: SqueezeLLM基于权重量化特性，结合了稠密和稀疏量化，提出了针对LLM特性的量化策略。
## ATOM
- 论文: ATOM: LOW-BIT QUANTIZATION FOR EFFICIENT AND ACCURATE LLM SERVING
- 代码: https://github.com/efeslab/Atom
- 类型: W4A4
- 描述: ATOM针对LLM特性，提出混合精度量化策略，包括对outlier activation的INT8量化，normal值的INT4量化，以及group量化和动态量化技术。
## OliVe
- 论文: OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization
- 类型: W4A4
- 描述: OliVe从硬件层面优化离群值计算，提出了牺牲正常值以适应离群值的量化策略，提高硬件效率。
## Outlier Suppression
- 论文: Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models
- 代码: https://github.com/wimh966/outlier_suppression
- 类型: W8A8
- 描述: Outlier Suppression通过抑制离群值放大器γ和对离群值进行剪切，来减少离群值对量化精度的影响。
