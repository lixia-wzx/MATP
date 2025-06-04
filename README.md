# MATP
## Model Architecture

![MATP](F:\研究生\研究生\MATP\MATP.png)
## Description

1. train.py: 训练文件
2. test.py:测试文件
3. model.py:模型文件
4. spatial_temporal.py:时空交互建模文件
5. map_encoder.py:层级注意力车道编码文件
6. PredNet.py:解码器文件
7. process_data_train.py:训练数据集处理文件
8. process_data_val.py:验证数据集处理文件
9. process_data_test.py:测试数据集处理文件

## Dataset

Argoverse数据集下载链接：[https://www.argoverse.org/av1.html](https://www.argoverse.org/av1.html)

## Training

`python -u train.py &`

## Testing

`python -u test.py &`
