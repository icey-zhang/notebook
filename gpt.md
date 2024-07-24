参考：[李沐论文精读】GPT、GPT-2和GPT-3论文精读](https://blog.csdn.net/qq_45276194/article/details/136530979)

# 实践项目 KAN-GPT-2
<img width="731" alt="image" src="https://github.com/user-attachments/assets/da087340-f21b-4c62-bb49-7be512327a9d">

[【Code】](https://github.com/CG80499/KAN-GPT-2)
[【Data】](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)

## step 1 环境安装
python3.8的话只能用optax==0.1.7，不然就会报错betas: tuple[float, float] = (0.9, 0.999), TypeError: 'type' object is not subscriptable
```python
pip install optax==0.1.7 -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

## step 2 下载数据集
[【数据集路径】](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)
修改数据集路径tiny_stories.py第13行
```python
dataset = load_dataset("/home/zjq/dataset/TinyStories")
```

## step 3 运行代码transformer.py
会出现错误
attn = nn.MultiHeadDotProductAttention(
TypeError: __call__() missing 1 required positional argument: 'inputs_kv'

修改第73行
```python
attn = nn.MultiHeadDotProductAttention(
    num_heads=n_heads, qkv_features=d_model // n_heads, out_features=d_model, param_dtype=D_TYPE
)(y,y, mask=mask)
```
运行成功

<img width="912" alt="image" src="https://github.com/user-attachments/assets/17e24bd5-773a-47ab-bb45-22f70fd66c7c">

项目原作者提供的wandb地址：[wandb](https://wandb.ai/cg123/kan-transformer?nw=nwusercg123)

## step 4 代码解读

### 完整代码

完整代码整合了上述所有部分，定义了一个自注意力块，执行层归一化、多头注意力并使用残差连接输出结果。

```python
import flax.linen as nn
import jax.numpy as jnp

D_TYPE = jnp.float32  # 例如，指定数据类型

class SelfAttentionBlock(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        # Shape (batch_size, seq_len, d_model)
        n_heads, d_model = self.n_heads, self.d_model
        assert d_model % n_heads == 0, 'n_heads must divide d_model'
        # Shape (batch_size, num_heads, seq_len, seq_len)
        mask = jnp.ones((x.shape[0], n_heads, x.shape[1], x.shape[1]))
        # Create diagonal mask
        mask = jnp.tril(mask)
        y = nn.LayerNorm(param_dtype=D_TYPE)(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=n_heads, qkv_features=d_model // n_heads, out_features=d_model, param_dtype=D_TYPE
        )(y, y, mask=mask)
        return x + attn
```

这个模块可用于自注意力机制，是 Transformer 架构的一部分。
这段代码定义了一个基于 Flax 库的自注意力块 (SelfAttentionBlock) 模块。这个模块使用多头点积注意力 (MultiHeadDotProductAttention) 来处理输入。下面对每一行代码进行详细解释：

### 模块定义和初始化参数

```python
class SelfAttentionBlock(nn.Module):
    d_model: int
    n_heads: int
```

- `class SelfAttentionBlock(nn.Module)`: 定义了一个继承自 `flax.linen.Module` 的自注意力块类。
- `d_model: int`: 声明模型的特征维度大小（特征的数量）。
- `n_heads: int`: 声明多头注意力的头数。

### 定义 `__call__` 方法

```python
@nn.compact
def __call__(self, x):
```

- `@nn.compact`: 这是一个 Flax 装饰器，表明这个方法内部定义的层（如注意力层、层归一化层等）是 "紧凑" 的，Flax 将自动处理它们的参数。
- `def __call__(self, x)`: 定义了模块的前向传递方法，`x` 是输入张量。

### 参数和断言

```python
# Shape (batch_size, seq_len, d_model)
n_heads, d_model = self.n_heads, self.d_model
assert d_model % n_heads == 0, 'n_heads must divide d_model'
```

- `n_heads, d_model = self.n_heads, self.d_model`: 获取类初始化时传入的 `n_heads` 和 `d_model`。
- `assert d_model % n_heads == 0, 'n_heads must divide d_model'`: 确保 `d_model` 可以被 `n_heads` 整除，因为每个注意力头将处理 `d_model // n_heads` 维度。

### 创建掩码

```python
# Shape (batch_size, num_heads, seq_len, seq_len)
mask = jnp.ones((x.shape[0], n_heads, x.shape[1], x.shape[1]))
# Create diagonal mask
mask = jnp.tril(mask)
```

- `mask = jnp.ones((x.shape[0], n_heads, x.shape[1], x.shape[1]))`: 创建一个形状为 `(batch_size, n_heads, seq_len, seq_len)` 的全 1 掩码张量，用于多头注意力。
- `mask = jnp.tril(mask)`: 将掩码转化为下三角矩阵，确保在自回归模型中仅关注先前位置（防止未来信息泄露）。

### 层归一化

```python
y = nn.LayerNorm(param_dtype=D_TYPE)(x)
```

- `y = nn.LayerNorm(param_dtype=D_TYPE)(x)`: 对输入 `x` 进行层归一化，`param_dtype=D_TYPE` 指定参数数据类型。`LayerNorm` 层使每个样本在批次中的均值为 0，方差为 1，稳定训练。

### 多头注意力

```python
attn = nn.MultiHeadDotProductAttention(
    num_heads=n_heads, qkv_features=d_model // n_heads, out_features=d_model, param_dtype=D_TYPE
)(y, y, mask=mask)
```

- `attn = nn.MultiHeadDotProductAttention(...)`: 初始化 `MultiHeadDotProductAttention` 层，并传入参数：
  - `num_heads=n_heads`: 注意力头数。
  - `qkv_features=d_model // n_heads`: 每个注意力头的特征维度。
  - `out_features=d_model`: 输出特征维度。
  - `param_dtype=D_TYPE`: 参数数据类型。
- `(...)(y, y, mask=mask)`: 调用注意力层，传入 `y` 作为查询、键和值，以及之前创建的掩码 `mask`。

### 残差连接

```python
return x + attn
```

- `return x + attn`: 将注意力输出 `attn` 与输入 `x` 相加，实现残差连接，帮助梯度传播并稳定训练。

## step 5 修改代码
要把自注意力机制改成**可变形卷积**

[【卷积参考】](https://www.bilibili.com/video/BV1Sh4y1y75i/?spm_id_from=333.337.search-card.all.click&vd_source=ee28f748a7042b99cf81403720f8106e)

**普通卷积**

<img width="600" alt="image" src="https://github.com/user-attachments/assets/bb610fdc-3316-46d4-81fa-a2379cc7ff1b">

**空洞卷积**

<img width="600" alt="image" src="https://github.com/user-attachments/assets/974c7acd-d13a-45a0-9f0b-59304d71a5e4">

**Depth-wise 卷积**
[【Conv2Former】](https://github.com/HVision-NKU/Conv2Former/blob/main/convmod.py)
这篇论文的[【讲解】](https://zhuanlan.zhihu.com/p/589738842)
卷积的[【解释】](https://blog.csdn.net/Bolly_He/article/details/124107316)
概念

Depth-wise 卷积是一种高效的卷积操作，主要用于降低计算复杂度和参数数量。它将传统卷积分解为两个步骤：depth-wise 卷积和 point-wise 卷积。

步骤
1.Depth-wise 卷积: 对输入的每个通道分别进行卷积操作。假设输入有C个通道，每个通道使用一个单独的$3\times 3$卷积核进行卷积，因此总共有C个卷积核。
2.Point-wise 卷积: 使用$1\times 1$卷积对 depth-wise 卷积的输出进行线性组合，将通道信息重新组合。

优点
- 减少了计算量和参数数量。
- 适用于移动设备和资源受限的场景。
 
**可变形卷积Deformable Convolutional Networks**

<img width="600" alt="image" src="https://github.com/user-attachments/assets/75fa9dbb-5717-45d4-b29c-55667190b89c">

有9个偏移量，每个偏移量有x，y两个参数，所以通道数为18

<img width="600" alt="image" src="https://github.com/user-attachments/assets/771e4853-4d10-4d77-b5bb-1ba54150dddc">

得到亚像素点的位置

<img width="300" alt="image" src="https://github.com/user-attachments/assets/2494608f-936e-4dfb-9d42-7be31c6ecdd9">

亚像素点位置的像素指再和w算卷积结果

<img width="600" alt="image" src="https://github.com/user-attachments/assets/5ae0d34e-f953-43c2-a815-bd104f4d709f">

怎么得到亚像素点位置的像素值-双线性插值

<img width="600" alt="image" src="https://github.com/user-attachments/assets/3599569d-730d-4857-940e-fed0e9ca78a2">


概念
可变形卷积通过引入**可学习的偏移量**，使卷积核的位置可以动态调整，从而增强模型处理几何变形的能力。这种操作在处理物体检测和分割任务中非常有用，因为它能够适应物体形状的变化。

步骤
1.偏移量计算: 使用一个卷积层计算每个位置的偏移量。
2.应用偏移量: 根据计算出的偏移量调整卷积核的位置，进行卷积操作。

优点
- 提高了模型在处理几何变形上的鲁棒性。
- 适用于物体检测和分割等需要处理复杂几何变形的任务。

### **如何修改？**
#### 定义卷积模块
```python
class ConvMod(nn.Module):
    dim: int
    kernel_size: int = 11  # 卷积核的大小
    padding: int = 5       # 填充大小

    @nn.compact
    def __call__(self, x):
        # 先对输入进行 LayerNorm
        x = nn.LayerNorm(epsilon=1e-6)(x)

        # 一维卷积层
        conv_a = nn.Conv(features=self.dim, kernel_size=(1,))(x)  # 1D卷积，改变通道数
        conv_a = jax.nn.gelu(conv_a)  # 使用 jax.nn.gelu 进行激活
        # print(conv_a.shape) #(16, 64, 128)
        conv_a = nn.Conv(features=self.dim, kernel_size=(self.kernel_size,), padding='SAME', feature_group_count=self.dim)(conv_a)  # depthwise 1D卷积
        conv_v = nn.Conv(features=self.dim, kernel_size=(1,))(x)  # 1D卷积，用于生成卷积的值
        x = conv_a * conv_v  # 逐元素相乘
        x = nn.Conv(features=self.dim, kernel_size=(1,))(x)  # 1D卷积，映射回原始维度

        return x

```
#### 替换自注意力机制 
```python
class KANTransformer(nn.Module):
    d_model: int
    n_heads: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        # Shape (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = nn.Embed(num_embeddings=TOKENIZER_SIZE, features=self.d_model, param_dtype=D_TYPE)(x)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=self.d_model, param_dtype=D_TYPE)(jnp.arange(MAX_LEN))
        x = x + pos_emb
        for _ in range(self.n_layers):
            #### 修改注意力机制为可变卷积 ###
            # print(x.shape) #(16, 64, 128)
            x = ConvMod(self.d_model, self.n_heads)(x)
            x = KANBlock()(x)
        # Shape (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        x = nn.Dense(features=TOKENIZER_SIZE, use_bias=False, param_dtype=D_TYPE)(x)
        return x
```

训练4h 47m 5s，一半报错 Segmentation fault









