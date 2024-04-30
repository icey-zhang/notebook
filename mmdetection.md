关于在使用mmdetection时遇到的一些问题
## data['category_id'] = self.cat_ids[label]
IndexError: list index out of range

找到相关配置文件,修改其中的 conf_thres 阈值
```bash
比如我是
vi configs/base/models/faster_rcnn_r50_fpn.py ,打开配置文件；
修改 test_cfg 里的
score_thr=0.3
```
https://blog.csdn.net/qq_36810398/article/details/116994577
上面那个方法没有用
https://blog.csdn.net/qq_38018994/article/details/122320888

## RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

因为显存不够，改小batchsize即可

## python setup.py install/build 与 python setup.py develop的区别 (python setup.py build_ext --inplace)
可以用
python setup.py --help-commands
查看help

      在源码安装某个库或包时，发现可以python setup.py install 和python setup.py develop两种方式来安装，这两种方法的区别是

①python setup.py install：主要是安装典型第三方包，这种包比较稳定，不再需要你去编辑、修改或是调试。

②python setup.py develop:当你安装一个包后，这个包需要你不断修改，这样你就不得不重新安装，这时就采用这种安装方法。

当执行python setup.py install, 程序做的事情很简单，就是 copy build/lib(或build/lib.plat)目录下的everything到python安装目录。linux下的安装目录通常是prefix/lib/pythonX.Y/site-packages


![image](https://github.com/icey-zhang/notebook/assets/54712081/6d407603-cd11-4ce5-b71f-ef26c12dd155)


而python setup.py develop不会真正安装包，而是在系统环境中创建一个软连接指向包实际所在的目录，这样修改了相关文件之后不用再安装便能生效，便于开发调试等

有时候我们会看到有人用python setup.py build命令，有些困惑

其实python setup.py install 一条命令就已经把build 和 install都做了，但是也可以将二者分开，就像比如你想在本地build好，然后发给其他人直接install就行了，就可以这样
```python
python setup.py build
python setup.py install
```
也有人这么写

```python
python setup.py build install
```
这就和python setup.py install没什么区别了

build的作用是

![image](https://github.com/icey-zhang/notebook/assets/54712081/91570025-ecb6-401c-9086-a685669c5013)


即如果你的package中有C文件，那么他们会同时被编译，否则build命令做的就是copying

python setup.py build_ext --inplace
build_ext:build C/C++ extensions (compile/link to build directory)，给python编译一个c、c++的拓展
–inplace:ignore build-lib and put compiled extensions into the source directory alongside your pure Python modules，忽略build-lib，将编译后的扩展放到源目录中，与纯Python模块放在一起

## 安装MultiScaleDeformableAttention不成功
可以把下面import的代码修改一下
```python
# import MultiScaleDeformableAttention as MSDA
import mmcv.ops.multi_scale_deform_attn as MSDA
```

## 一些包换路径了
https://mmengine.readthedocs.io/zh-cn/v0.3.0/api/generated/mmengine.hooks.Hook.html?highlight=HOOKS

from mmengine.runner import load_checkpoint
