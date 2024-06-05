在神经网络中，常用 From thop import profile 来计算FLOPs和Parameters来作为神经网络模型的评价指标。我在使用该函数时程序报如下错误：RuntimeError: module must have its parameters and buffers on device cuda:0 (device_ids[0]) but found one of them on device: cpu
https://blog.csdn.net/m0_48937452/article/details/126706620
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
 
input = torch.randn(1, 8, 224, 224).to(device)
flops, params = profile(model, inputs = (input, ))
```
改成
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
 
input = torch.randn(1, 8, 224, 224).to(device)
flops, params = profile(model.module, inputs = (input, ))
```

检查模型参数和缓存所在的device
```python
for param in model.parameters():
    print(param.device)

for buffer in model.buffers():
    print(buffer.device)
```
