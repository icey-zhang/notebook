参考：[李沐论文精读】GPT、GPT-2和GPT-3论文精读](https://blog.csdn.net/qq_45276194/article/details/136530979)

# 实践项目 KAN-GPT-2
[【Code】](https://github.com/CG80499/KAN-GPT-2)
[【Data】](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)

## step 1 环境安装
python3.8的话只能用optax==0.1.7，不然就会报错betas: tuple[float, float] = (0.9, 0.999), TypeError: 'type' object is not subscriptable
```python
pip install optax==0.1.7 -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

## step 2 下载数据集
[【数据集路径】](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)

## step 3 运行代码transfomer.py
会出现错误
attn = nn.MultiHeadDotProductAttention(
TypeError: __call__() missing 1 required positional argument: 'inputs_kv'

修改第73行
```python
attn = nn.MultiHeadDotProductAttention(
    num_heads=n_heads, qkv_features=d_model // n_heads, out_features=d_model, param_dtype=D_TYPE
)(y,y, mask=mask)
```





