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
