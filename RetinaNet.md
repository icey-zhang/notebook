1.前言
RetinaNet是继SSD和YOLO V2公布后，YOLO V3诞生前的一款目标检测模型，出自何恺明大神的《Focal Loss for Dense Object Detection》。全文针对现有单阶段法（one-stage)目标检测模型中前景(positive)和背景(negatives)类别的不平衡问题，提出了一种叫做Focal Loss的损失函数，用来降低大量easy negatives在标准交叉熵中所占权重（提高hard negatives所占权重)。为了检测提出的Focal Loss损失函数的有效性，所以作者就顺便提出了一种简单的模型RetinaNet。（所以RetinaNet不是本篇论文的主角，仅仅是附属物了呗？）
