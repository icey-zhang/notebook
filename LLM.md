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
GPTQ
论文: GPTQ: 准确的训练后量化用于生成预训练变换器
代码: GitHub
类型: W4A16
特点: 利用Hessian矩阵进行权重更新，加速量化过程，舍弃贪心策略，使用批处理更新和Cholesky分解。
LLM.int8()
论文: LLM.int8: 大规模变换器的8位矩阵乘法
代码: GitHub
类型: W8A8
特点: 发现并处理离群点，采用分而治之策略，混合精度计算。
SmoothQuant
论文: SmoothQuant: 大语言模型准确高效的训练后量化
代码: GitHub
类型: W8A8
特点: 通过平滑权重和激活值，解决量化难点，已集成到主流框架中。
AWQ
论文: AWQ: 激活感知的权重量化用于LLM压缩和加速
代码: GitHub
类型: W4A16
特点: 关注低比特量化，发现权重敏感性不平衡，使用激活值定位关键权重，保护模型性能。
ZeroQuant系列
代码: GitHub
特点: 提供端到端量化推理流水线，涵盖多种量化方法，适用于大规模模型。
Olive
类型: W4A4
特点: 从硬件层面优化离群值计算，局部处理离群值，提高硬件效率。
Outlier Suppression
论文: Outlier Suppression: 推动低比特变换器语言模型的极限
代码: GitHub
类型: W8A8
特点: 抑制离群值影响，通过Gamma迁移和token-wise剪切优化量化。
SpQR
论文: SpQR: 近无损的LLM权重压缩的稀疏量化表示
特点: 结合AWQ、GPTQ和LLM.int8()，识别敏感权重，提供混合精度压缩表示。
OWQ
论文: OWQ: 异常值感知的权重量化用于LLM有效微调和推理
代码: GitHub
类型: W4A16
特点: 使用海森矩阵筛选敏感列，存储在更高精度中，支持特定任务的微调。
SqueezeLLM
代码: GitHub
类型: W4A16
特点: 使用近似Fisher信息度量敏感度，非均匀量化，稠密和稀疏分解。
RPTQ
论文: RPTQ: 基于重排的大型语言模型训练后量化
代码: GitHub
类型: W4A4/W4A8/W4A4KV
特点: 重排激活值channel，减少量化误差，优化内存重排开销。
ATOM
论文: ATOM: 低比特量化用于高效和准确的LLM服务
代码: GitHub
请注意，以上信息是根据提供的链接内容整理而成，具体细节和完整算法描述请参阅原论文或代码仓库。
