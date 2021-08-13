# [AI训练营]paddleclas实现图像分类baseline
采用paddleclas套件实现分类，玩转PaddleClas
# 一、项目背景
本项目属于本次课程的大作业的一部分，可以学会使用paddleclas实现图像分类。
# 二、数据集简介
<font size="3px" color="blue">本次数据集有五个。分别是：</font>   
1. 猫12分类
2. 垃圾40分类
3. 场景5分类
4. 食物5分类
5. 蝴蝶20分类

**数据集：都是不同类别的文件夹下放置了对应文件夹名字的类别图片**



# 三、实现过程


```python
# 先导入库
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
import random
```


```python
# 忽略（垃圾）警告信息
# 在python中运行代码经常会遇到的情况是——代码可以正常运行但是会提示警告，有时特别讨厌。
# 那么如何来控制警告输出呢？其实很简单，python通过调用warnings模块中定义的warn()函数来发出警告。我们可以通过警告过滤器进行控制是否发出警告消息。
import warnings
warnings.filterwarnings("ignore")
```

### 3.1 解压数据集，查看数据的结构


```python
# 项目挂载的数据集先解压出来，待解压完毕，刷新后可发现左侧文件夹根目录出现五个zip
!unzip -oq /home/aistudio/data/data103736/五种图像分类数据集.zip
```

左侧可以看到如图所示五个zip    
![](https://ai-studio-static-online.cdn.bcebos.com/f8bc5b21a0ba49b4b78b6e7b18ac0341dfb14cf545b14c83b1f597b6ee8109bb)



```python
# 本项目以食物分类为例进行介绍，因为分类大多数情况下是不存在标签文件的，猫分类已经有了标签，省去了数据处理的操作
# (此处需要你根据自己的选择进行解压对应的文件)
# 解压完毕左侧出现文件夹，即为需要分类的文件
!unzip -oq /home/aistudio/食物5分类.zip
```


```python
# 查看结构，正为一个类别下有一系列对应的图片
!tree foods/
```

    5 directories, 5000 files

**五类食物图片**  
1. beef_carpaccio
2. baby_back_ribs
3. beef_tartare
4. apple_pie
5. baklava    

具体结构如下：
```
foods/
├── apple_pie
│   ├── 1005649.jpg
│   ├── 1011328.jpg
│   ├── 101251.jpg
```

### 3.2 拿到总的训练数据txt


```python
import os
# -*- coding: utf-8 -*-
# 根据官方paddleclas的提示，我们需要把图像变为两个txt文件
# train_list.txt（训练集）
# val_list.txt（验证集）
# 先把路径搞定 比如：foods/beef_carpaccio/855780.jpg ,读取到并写入txt 

# 根据左侧生成的文件夹名字来写根目录
dirpath = "foods"
# 先得到总的txt后续再进行划分，因为要划分出验证集，所以要先打乱，因为原本是有序的
def get_all_txt():
    all_list = []
    i = 0 # 标记总文件数量
    j = 0 # 标记文件类别
    for root,dirs,files in os.walk(dirpath): # 分别代表根目录、文件夹、文件
        for file in files:
            i = i + 1 
            # 文件中每行格式： 图像相对路径      图像的label_id（数字类别）（注意：中间有空格）。              
            imgpath = os.path.join(root,file)
            all_list.append(imgpath+" "+str(j)+"\n")

        j = j + 1

    allstr = ''.join(all_list)
    f = open('all_list.txt','w',encoding='utf-8')
    f.write(allstr)
    return all_list , i
all_list,all_lenth = get_all_txt()
print(all_lenth)
```

    5000


### 3.3 数据打乱


```python
# 把数据打乱
all_list = shuffle(all_list)
allstr = ''.join(all_list)
f = open('all_list.txt','w',encoding='utf-8')
f.write(allstr)
print("打乱成功，并重新写入文本")
```

    打乱成功，并重新写入文本


### 3.4 数据划分


```python
# 按照比例划分数据集 食品的数据有5000张图片，不算大数据，一般9:1即可
train_size = int(all_lenth * 0.9)
train_list = all_list[:train_size]
val_list = all_list[train_size:]

print(len(train_list))
print(len(val_list))
```

    4500
    500



```python
# 运行cell，生成训练集txt 
train_txt = ''.join(train_list)
f_train = open('train_list.txt','w',encoding='utf-8')
f_train.write(train_txt)
f_train.close()
print("train_list.txt 生成成功！")

# 运行cell，生成验证集txt
val_txt = ''.join(val_list)
f_val = open('val_list.txt','w',encoding='utf-8')
f_val.write(val_txt)
f_val.close()
print("val_list.txt 生成成功！")
```

    train_list.txt 生成成功！
    val_list.txt 生成成功！


## 3.4 安装paddleclas

数据集核实完搞定成功的前提下，可以准备更改原文档的参数进行实现自己的图片分类了！

这里采用paddleclas的2.2版本，好用！


```python
# 先把paddleclas安装上再说
# 安装paddleclas以及相关三方包(好像studio自带的已经够用了，无需安装了)
!git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.2
# 我这里安装相关包时，花了30几分钟还有错误提示，不管他即可
#!pip install --upgrade -r PaddleClas/requirements.txt -i https://mirror.baidu.com/pypi/simple
```

    Cloning into 'PaddleClas'...
    remote: Enumerating objects: 538, done.[K
    remote: Counting objects: 100% (538/538), done.[K
    remote: Compressing objects: 100% (323/323), done.[K
    remote: Total 15290 (delta 347), reused 349 (delta 210), pack-reused 14752[K
    Receiving objects: 100% (15290/15290), 113.56 MiB | 12.99 MiB/s, done.
    Resolving deltas: 100% (10239/10239), done.
    Checking connectivity... done.



```python
#因为后续paddleclas的命令需要在PaddleClas目录下，所以进入PaddleClas根目录，执行此命令
%cd PaddleClas
!ls
```

    /home/aistudio/PaddleClas
    dataset  hubconf.py   MANIFEST.in    README_ch.md  requirements.txt
    deploy	 __init__.py  paddleclas.py  README_en.md  setup.py
    docs	 LICENSE      ppcls	     README.md	   tools



```python
# 将图片移动到paddleclas下面的数据集里面
# 至于为什么现在移动，也是我的一点小技巧，防止之前移动的话，生成的txt的路径是全路径，反而需要去掉路径的一部分
!mv ../foods/ dataset/
```


```python
# 挪动文件到对应目录
!mv ../all_list.txt dataset/foods
!mv ../train_list.txt dataset/foods
!mv ../val_list.txt dataset/foods
```


### 3.5 修改配置文件
#### 3.5.1
主要是以下几点：分类数、图片总量、训练和验证的路径、图像尺寸、数据预处理、训练和预测的num_workers: 0

路径如下：
>PaddleClas/ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml

<font size="3px" color="red">（将下列代码复制到上述路径的文件ShuffleNetV2_x0_25.yaml中）</font>

```
# global configs
Global:
  checkpoints: null
  pretrained_model: null
  output_dir: ./output/
  # 使用CPU训练
  device: cpu
  # 每几个轮次保存一次
  save_interval: 1 
  eval_during_train: True
  # 每几个轮次验证一次
  eval_interval: 1 
  # 训练轮次
  epochs: 2
  print_batch_step: 1
  use_visualdl: True #开启可视化（目前平台不可用）
  # used for static mode and model export
  # 图像大小
  image_shape: [3, 224, 224] 
  save_inference_dir: ./inference
  # training model under @to_static
  to_static: False

# model architecture
Arch:
  # 采用的网络
  name: ResNet50
  # 类别数 多了个0类 0-5 0无用 
  class_num: 6 
 
# loss function config for traing/eval process
Loss:
  Train:

    - CELoss: 
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0


Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Piecewise
    learning_rate: 0.015
    decay_epochs: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
  regularizer:
    name: 'L2'
    coeff: 0.0005


# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的训练集文本路径
      cls_label_path: ./dataset/foods/train_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

  Eval:
    dataset: 
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的验证集文本路径
      cls_label_path: ./dataset/foods/val_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

Infer:
  infer_imgs: ./dataset/foods/beef_carpaccio/855780.jpg
  batch_size: 10
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 256
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    # 输出的可能性最高的前topk个
    topk: 5
    # 标签文件 需要自己新建文件
    class_id_map_file: ./dataset/label_list.txt

Metric:
  Train:
    - TopkAcc:
        topk: [1, 5]
  Eval:
    - TopkAcc:
        topk: [1, 5]
```
#### 3.5.2 标签文件
这个是在预测时生成对照的依据，在上个文件有提到这个
```
# 标签文件 需要自己新建文件
    class_id_map_file: dataset/label_list.txt
```
按照对应的进行编写：   

![](https://ai-studio-static-online.cdn.bcebos.com/0e40a0afaa824ba9b70778aa7931a3baf2a421bcb81b4b0f83632da4e4ddc0ef)  

如食品分类(要对照之前的txt的类别确认无误) 
```
1 beef_carpaccio
2 baby_back_ribs
3 beef_tartare
4 apple_pie
5 baklava
```
![](https://ai-studio-static-online.cdn.bcebos.com/6b3a4a244ed34517bcf4be5fcc629ab10d081eb0af7c4532a795583d19939a82)


![](https://ai-studio-static-online.cdn.bcebos.com/8ee69dbad8bf4e0f9c743645a4438cc035446d052d4b409a88c1cc25b58dcf83)

### 3.6 开始训练


```python
# 提示，运行过程中可能存在坏图的情况，但是不用担心，训练过程不受影响。
# 仅供参考，我只跑了五轮，准确率很低
!python3 tools/train.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml
```

    /home/aistudio/PaddleClas/ppcls/arch/backbone/model_zoo/vision_transformer.py:15: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Callable
    [2021/08/13 19:35:52] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2021/08/13 19:35:52] root INFO: Arch : 
    [2021/08/13 19:35:52] root INFO:     class_num : 6
    [2021/08/13 19:35:52] root INFO:     name : ResNet50
    [2021/08/13 19:35:52] root INFO: DataLoader : 
    [2021/08/13 19:35:52] root INFO:     Eval : 
    [2021/08/13 19:35:52] root INFO:         dataset : 
    [2021/08/13 19:35:52] root INFO:             cls_label_path : ./dataset/foods/val_list.txt
    [2021/08/13 19:35:52] root INFO:             image_root : ./dataset/
    [2021/08/13 19:35:52] root INFO:             name : ImageNetDataset
    [2021/08/13 19:35:52] root INFO:             transform_ops : 
    [2021/08/13 19:35:52] root INFO:                 DecodeImage : 
    [2021/08/13 19:35:52] root INFO:                     channel_first : False
    [2021/08/13 19:35:52] root INFO:                     to_rgb : True
    [2021/08/13 19:35:52] root INFO:                 ResizeImage : 
    [2021/08/13 19:35:52] root INFO:                     resize_short : 256
    [2021/08/13 19:35:52] root INFO:                 CropImage : 
    [2021/08/13 19:35:52] root INFO:                     size : 224
    [2021/08/13 19:35:52] root INFO:                 NormalizeImage : 
    [2021/08/13 19:35:52] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/13 19:35:52] root INFO:                     order : 
    [2021/08/13 19:35:52] root INFO:                     scale : 1.0/255.0
    [2021/08/13 19:35:52] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/13 19:35:52] root INFO:         loader : 
    [2021/08/13 19:35:52] root INFO:             num_workers : 0
    [2021/08/13 19:35:52] root INFO:             use_shared_memory : True
    [2021/08/13 19:35:52] root INFO:         sampler : 
    [2021/08/13 19:35:52] root INFO:             batch_size : 128
    [2021/08/13 19:35:52] root INFO:             drop_last : False
    [2021/08/13 19:35:52] root INFO:             name : DistributedBatchSampler
    [2021/08/13 19:35:52] root INFO:             shuffle : True
    [2021/08/13 19:35:52] root INFO:     Train : 
    [2021/08/13 19:35:52] root INFO:         dataset : 
    [2021/08/13 19:35:52] root INFO:             cls_label_path : ./dataset/foods/train_list.txt
    [2021/08/13 19:35:52] root INFO:             image_root : ./dataset/
    [2021/08/13 19:35:52] root INFO:             name : ImageNetDataset
    [2021/08/13 19:35:52] root INFO:             transform_ops : 
    [2021/08/13 19:35:52] root INFO:                 DecodeImage : 
    [2021/08/13 19:35:52] root INFO:                     channel_first : False
    [2021/08/13 19:35:52] root INFO:                     to_rgb : True
    [2021/08/13 19:35:52] root INFO:                 ResizeImage : 
    [2021/08/13 19:35:52] root INFO:                     resize_short : 256
    [2021/08/13 19:35:52] root INFO:                 CropImage : 
    [2021/08/13 19:35:52] root INFO:                     size : 224
    [2021/08/13 19:35:52] root INFO:                 RandFlipImage : 
    [2021/08/13 19:35:52] root INFO:                     flip_code : 1
    [2021/08/13 19:35:52] root INFO:                 NormalizeImage : 
    [2021/08/13 19:35:52] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/13 19:35:52] root INFO:                     order : 
    [2021/08/13 19:35:52] root INFO:                     scale : 1.0/255.0
    [2021/08/13 19:35:52] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/13 19:35:52] root INFO:         loader : 
    [2021/08/13 19:35:52] root INFO:             num_workers : 0
    [2021/08/13 19:35:52] root INFO:             use_shared_memory : True
    [2021/08/13 19:35:52] root INFO:         sampler : 
    [2021/08/13 19:35:52] root INFO:             batch_size : 128
    [2021/08/13 19:35:52] root INFO:             drop_last : False
    [2021/08/13 19:35:52] root INFO:             name : DistributedBatchSampler
    [2021/08/13 19:35:52] root INFO:             shuffle : True
    [2021/08/13 19:35:52] root INFO: Global : 
    [2021/08/13 19:35:52] root INFO:     checkpoints : None
    [2021/08/13 19:35:52] root INFO:     device : cpu
    [2021/08/13 19:35:52] root INFO:     epochs : 2
    [2021/08/13 19:35:52] root INFO:     eval_during_train : True
    [2021/08/13 19:35:52] root INFO:     eval_interval : 1
    [2021/08/13 19:35:52] root INFO:     image_shape : [3, 224, 224]
    [2021/08/13 19:35:52] root INFO:     output_dir : ./output/
    [2021/08/13 19:35:52] root INFO:     pretrained_model : None
    [2021/08/13 19:35:52] root INFO:     print_batch_step : 1
    [2021/08/13 19:35:52] root INFO:     save_inference_dir : ./inference
    [2021/08/13 19:35:52] root INFO:     save_interval : 1
    [2021/08/13 19:35:52] root INFO:     to_static : False
    [2021/08/13 19:35:52] root INFO:     use_visualdl : True
    [2021/08/13 19:35:52] root INFO: Infer : 
    [2021/08/13 19:35:52] root INFO:     PostProcess : 
    [2021/08/13 19:35:52] root INFO:         class_id_map_file : ./dataset/label_list.txt
    [2021/08/13 19:35:52] root INFO:         name : Topk
    [2021/08/13 19:35:52] root INFO:         topk : 5
    [2021/08/13 19:35:52] root INFO:     batch_size : 10
    [2021/08/13 19:35:52] root INFO:     infer_imgs : ./dataset/foods/beef_carpaccio/855780.jpg
    [2021/08/13 19:35:52] root INFO:     transforms : 
    [2021/08/13 19:35:52] root INFO:         DecodeImage : 
    [2021/08/13 19:35:52] root INFO:             channel_first : False
    [2021/08/13 19:35:52] root INFO:             to_rgb : True
    [2021/08/13 19:35:52] root INFO:         ResizeImage : 
    [2021/08/13 19:35:52] root INFO:             resize_short : 256
    [2021/08/13 19:35:52] root INFO:         CropImage : 
    [2021/08/13 19:35:52] root INFO:             size : 224
    [2021/08/13 19:35:52] root INFO:         NormalizeImage : 
    [2021/08/13 19:35:52] root INFO:             mean : [0.485, 0.456, 0.406]
    [2021/08/13 19:35:52] root INFO:             order : 
    [2021/08/13 19:35:52] root INFO:             scale : 1.0/255.0
    [2021/08/13 19:35:52] root INFO:             std : [0.229, 0.224, 0.225]
    [2021/08/13 19:35:52] root INFO:         ToCHWImage : None
    [2021/08/13 19:35:52] root INFO: Loss : 
    [2021/08/13 19:35:52] root INFO:     Eval : 
    [2021/08/13 19:35:52] root INFO:         CELoss : 
    [2021/08/13 19:35:52] root INFO:             weight : 1.0
    [2021/08/13 19:35:52] root INFO:     Train : 
    [2021/08/13 19:35:52] root INFO:         CELoss : 
    [2021/08/13 19:35:52] root INFO:             weight : 1.0
    [2021/08/13 19:35:52] root INFO: Metric : 
    [2021/08/13 19:35:52] root INFO:     Eval : 
    [2021/08/13 19:35:52] root INFO:         TopkAcc : 
    [2021/08/13 19:35:52] root INFO:             topk : [1, 5]
    [2021/08/13 19:35:52] root INFO:     Train : 
    [2021/08/13 19:35:52] root INFO:         TopkAcc : 
    [2021/08/13 19:35:52] root INFO:             topk : [1, 5]
    [2021/08/13 19:35:52] root INFO: Optimizer : 
    [2021/08/13 19:35:52] root INFO:     lr : 
    [2021/08/13 19:35:52] root INFO:         decay_epochs : [30, 60, 90]
    [2021/08/13 19:35:52] root INFO:         learning_rate : 0.015
    [2021/08/13 19:35:52] root INFO:         name : Piecewise
    [2021/08/13 19:35:52] root INFO:         values : [0.1, 0.01, 0.001, 0.0001]
    [2021/08/13 19:35:52] root INFO:     momentum : 0.9
    [2021/08/13 19:35:52] root INFO:     name : Momentum
    [2021/08/13 19:35:52] root INFO:     regularizer : 
    [2021/08/13 19:35:52] root INFO:         coeff : 0.0005
    [2021/08/13 19:35:52] root INFO:         name : L2
    [2021/08/13 19:35:53] root INFO: train with paddle 2.1.2 and device CPUPlace
    {'CELoss': {'weight': 1.0}}
    [2021/08/13 19:36:56] root INFO: [Train][Epoch 1/2][Iter: 0/36]lr: 0.10000, CELoss: 1.84147, loss: 1.84147, top1: 0.18750, top5: 0.80469, batch_cost: 62.87839s, reader_cost: 0.95061, ips: 2.03568 images/sec, eta: 1:15:27
    [2021/08/13 19:37:53] root INFO: [Train][Epoch 1/2][Iter: 1/36]lr: 0.10000, CELoss: 4.59667, loss: 4.59667, top1: 0.20312, top5: 0.90234, batch_cost: 60.05370s, reader_cost: 0.47542, ips: 2.13143 images/sec, eta: 1:11:03
    [2021/08/13 19:38:50] root INFO: [Train][Epoch 1/2][Iter: 2/36]lr: 0.10000, CELoss: 8.86604, loss: 8.86604, top1: 0.20052, top5: 0.93490, batch_cost: 59.08704s, reader_cost: 0.31702, ips: 2.16630 images/sec, eta: 1:08:56
    [2021/08/13 19:39:47] root INFO: [Train][Epoch 1/2][Iter: 3/36]lr: 0.10000, CELoss: 13.91719, loss: 13.91719, top1: 0.18945, top5: 0.94531, batch_cost: 58.58398s, reader_cost: 0.23782, ips: 2.18490 images/sec, eta: 1:07:22
    [2021/08/13 19:40:44] root INFO: [Train][Epoch 1/2][Iter: 4/36]lr: 0.10000, CELoss: 14.62479, loss: 14.62479, top1: 0.18594, top5: 0.95625, batch_cost: 58.26786s, reader_cost: 0.19030, ips: 2.19675 images/sec, eta: 1:06:02
    [2021/08/13 19:41:42] root INFO: [Train][Epoch 1/2][Iter: 5/36]lr: 0.10000, CELoss: 14.55515, loss: 14.55515, top1: 0.17969, top5: 0.96354, batch_cost: 57.13040s, reader_cost: 0.00023, ips: 2.24049 images/sec, eta: 1:03:47
    [2021/08/13 19:42:39] root INFO: [Train][Epoch 1/2][Iter: 6/36]lr: 0.10000, CELoss: 13.46807, loss: 13.46807, top1: 0.17969, top5: 0.96875, batch_cost: 57.04443s, reader_cost: 0.00023, ips: 2.24386 images/sec, eta: 1:02:44
    [2021/08/13 19:43:36] root INFO: [Train][Epoch 1/2][Iter: 7/36]lr: 0.10000, CELoss: 13.26998, loss: 13.26998, top1: 0.18262, top5: 0.97266, batch_cost: 57.00898s, reader_cost: 0.00022, ips: 2.24526 images/sec, eta: 1:01:45
    [2021/08/13 19:44:33] root INFO: [Train][Epoch 1/2][Iter: 8/36]lr: 0.10000, CELoss: 13.08553, loss: 13.08553, top1: 0.18576, top5: 0.97569, batch_cost: 57.01601s, reader_cost: 0.00022, ips: 2.24498 images/sec, eta: 1:00:49
    [2021/08/13 19:45:30] root INFO: [Train][Epoch 1/2][Iter: 9/36]lr: 0.10000, CELoss: 13.06020, loss: 13.06020, top1: 0.18594, top5: 0.97813, batch_cost: 57.02373s, reader_cost: 0.00022, ips: 2.24468 images/sec, eta: 0:59:52
    [2021/08/13 19:46:27] root INFO: [Train][Epoch 1/2][Iter: 10/36]lr: 0.10000, CELoss: 12.39369, loss: 12.39369, top1: 0.18821, top5: 0.98011, batch_cost: 57.03333s, reader_cost: 0.00022, ips: 2.24430 images/sec, eta: 0:58:56
    [2021/08/13 19:47:24] root INFO: [Train][Epoch 1/2][Iter: 11/36]lr: 0.10000, CELoss: 11.92468, loss: 11.92468, top1: 0.18685, top5: 0.98177, batch_cost: 57.09298s, reader_cost: 0.00022, ips: 2.24196 images/sec, eta: 0:58:02
    [2021/08/13 19:48:21] root INFO: [Train][Epoch 1/2][Iter: 12/36]lr: 0.10000, CELoss: 11.51015, loss: 11.51015, top1: 0.18750, top5: 0.98317, batch_cost: 57.09653s, reader_cost: 0.00022, ips: 2.24182 images/sec, eta: 0:57:05
    [2021/08/13 19:49:18] root INFO: [Train][Epoch 1/2][Iter: 13/36]lr: 0.10000, CELoss: 11.22830, loss: 11.22830, top1: 0.18806, top5: 0.98438, batch_cost: 57.10962s, reader_cost: 0.00022, ips: 2.24130 images/sec, eta: 0:56:09
    [2021/08/13 19:50:16] root INFO: [Train][Epoch 1/2][Iter: 14/36]lr: 0.10000, CELoss: 10.76723, loss: 10.76723, top1: 0.19010, top5: 0.98542, batch_cost: 57.10774s, reader_cost: 0.00022, ips: 2.24138 images/sec, eta: 0:55:12
    [2021/08/13 19:51:12] root INFO: [Train][Epoch 1/2][Iter: 15/36]lr: 0.10000, CELoss: 10.32947, loss: 10.32947, top1: 0.19141, top5: 0.98633, batch_cost: 57.08184s, reader_cost: 0.00022, ips: 2.24239 images/sec, eta: 0:54:13
    [2021/08/13 19:52:09] root INFO: [Train][Epoch 1/2][Iter: 16/36]lr: 0.10000, CELoss: 9.91328, loss: 9.91328, top1: 0.19026, top5: 0.98713, batch_cost: 57.07383s, reader_cost: 0.00022, ips: 2.24271 images/sec, eta: 0:53:16
    [2021/08/13 19:53:06] root INFO: [Train][Epoch 1/2][Iter: 17/36]lr: 0.10000, CELoss: 9.53234, loss: 9.53234, top1: 0.19444, top5: 0.98785, batch_cost: 57.07638s, reader_cost: 0.00022, ips: 2.24261 images/sec, eta: 0:52:19
    [2021/08/13 19:54:03] root INFO: [Train][Epoch 1/2][Iter: 18/36]lr: 0.10000, CELoss: 9.18797, loss: 9.18797, top1: 0.19202, top5: 0.98849, batch_cost: 57.05670s, reader_cost: 0.00022, ips: 2.24338 images/sec, eta: 0:51:21
    [2021/08/13 19:55:00] root INFO: [Train][Epoch 1/2][Iter: 19/36]lr: 0.10000, CELoss: 8.87879, loss: 8.87879, top1: 0.19297, top5: 0.98906, batch_cost: 57.05203s, reader_cost: 0.00022, ips: 2.24357 images/sec, eta: 0:50:23
    [2021/08/13 19:55:57] root INFO: [Train][Epoch 1/2][Iter: 20/36]lr: 0.10000, CELoss: 8.61545, loss: 8.61545, top1: 0.19271, top5: 0.98958, batch_cost: 57.04648s, reader_cost: 0.00022, ips: 2.24378 images/sec, eta: 0:49:26
    [2021/08/13 19:56:54] root INFO: [Train][Epoch 1/2][Iter: 21/36]lr: 0.10000, CELoss: 8.33976, loss: 8.33976, top1: 0.19105, top5: 0.99006, batch_cost: 57.05491s, reader_cost: 0.00022, ips: 2.24345 images/sec, eta: 0:48:29
    [2021/08/13 19:57:53] root INFO: [Train][Epoch 1/2][Iter: 22/36]lr: 0.10000, CELoss: 8.10350, loss: 8.10350, top1: 0.19260, top5: 0.99049, batch_cost: 57.13212s, reader_cost: 0.00022, ips: 2.24042 images/sec, eta: 0:47:36
    [2021/08/13 19:58:52] root INFO: [Train][Epoch 1/2][Iter: 23/36]lr: 0.10000, CELoss: 7.85357, loss: 7.85357, top1: 0.19043, top5: 0.99089, batch_cost: 57.21798s, reader_cost: 0.00022, ips: 2.23706 images/sec, eta: 0:46:43
    [2021/08/13 19:59:49] root INFO: [Train][Epoch 1/2][Iter: 24/36]lr: 0.10000, CELoss: 7.64624, loss: 7.64624, top1: 0.19187, top5: 0.99125, batch_cost: 57.20532s, reader_cost: 0.00061, ips: 2.23755 images/sec, eta: 0:45:45
    [2021/08/13 20:00:46] root INFO: [Train][Epoch 1/2][Iter: 25/36]lr: 0.10000, CELoss: 7.45635, loss: 7.45635, top1: 0.19081, top5: 0.99159, batch_cost: 57.19401s, reader_cost: 0.00059, ips: 2.23800 images/sec, eta: 0:44:48
    [2021/08/13 20:01:42] root INFO: [Train][Epoch 1/2][Iter: 26/36]lr: 0.10000, CELoss: 7.29505, loss: 7.29505, top1: 0.19126, top5: 0.99190, batch_cost: 57.17274s, reader_cost: 0.00058, ips: 2.23883 images/sec, eta: 0:43:49
    [2021/08/13 20:02:39] root INFO: [Train][Epoch 1/2][Iter: 27/36]lr: 0.10000, CELoss: 7.10686, loss: 7.10686, top1: 0.19029, top5: 0.99219, batch_cost: 57.16541s, reader_cost: 0.00056, ips: 2.23912 images/sec, eta: 0:42:52
    [2021/08/13 20:03:36] root INFO: [Train][Epoch 1/2][Iter: 28/36]lr: 0.10000, CELoss: 6.92275, loss: 6.92275, top1: 0.19181, top5: 0.99246, batch_cost: 57.15077s, reader_cost: 0.00055, ips: 2.23969 images/sec, eta: 0:41:54
    [2021/08/13 20:04:33] root INFO: [Train][Epoch 1/2][Iter: 29/36]lr: 0.10000, CELoss: 6.77103, loss: 6.77103, top1: 0.19297, top5: 0.99271, batch_cost: 57.13300s, reader_cost: 0.00053, ips: 2.24039 images/sec, eta: 0:40:56
    [2021/08/13 20:05:30] root INFO: [Train][Epoch 1/2][Iter: 30/36]lr: 0.10000, CELoss: 6.62101, loss: 6.62101, top1: 0.19380, top5: 0.99294, batch_cost: 57.12344s, reader_cost: 0.00052, ips: 2.24076 images/sec, eta: 0:39:59
    [2021/08/13 20:06:27] root INFO: [Train][Epoch 1/2][Iter: 31/36]lr: 0.10000, CELoss: 6.49395, loss: 6.49395, top1: 0.19287, top5: 0.99316, batch_cost: 57.13356s, reader_cost: 0.00051, ips: 2.24036 images/sec, eta: 0:39:02
    [2021/08/13 20:07:24] root INFO: [Train][Epoch 1/2][Iter: 32/36]lr: 0.10000, CELoss: 6.37667, loss: 6.37667, top1: 0.19366, top5: 0.99337, batch_cost: 57.13593s, reader_cost: 0.00050, ips: 2.24027 images/sec, eta: 0:38:05
    [2021/08/13 20:08:21] root INFO: [Train][Epoch 1/2][Iter: 33/36]lr: 0.10000, CELoss: 6.26488, loss: 6.26488, top1: 0.19301, top5: 0.99357, batch_cost: 57.13240s, reader_cost: 0.00049, ips: 2.24041 images/sec, eta: 0:37:08
    [2021/08/13 20:09:17] root INFO: [Train][Epoch 1/2][Iter: 34/36]lr: 0.10000, CELoss: 6.14758, loss: 6.14758, top1: 0.19353, top5: 0.99375, batch_cost: 57.07438s, reader_cost: 0.00048, ips: 2.24269 images/sec, eta: 0:36:08
    [2021/08/13 20:09:25] root INFO: [Train][Epoch 1/2][Iter: 35/36]lr: 0.10000, CELoss: 6.13431, loss: 6.13431, top1: 0.19400, top5: 0.99378, batch_cost: 55.51486s, reader_cost: 0.00047, ips: 0.36026 images/sec, eta: 0:34:14
    [2021/08/13 20:09:25] root INFO: [Train][Epoch 1/2][Avg]CELoss: 6.13431, loss: 6.13431, top1: 0.19400, top5: 0.99378
    {'CELoss': {'weight': 1.0}}
    [2021/08/13 20:09:44] root INFO: [Eval][Epoch 1][Iter: 0/4]CELoss: 49.05455, loss: 49.05455, top1: 0.19531, top5: 1.00000, batch_cost: 18.21300s, reader_cost: 0.81978, ips: 7.02795 images/sec
    [2021/08/13 20:10:01] root INFO: [Eval][Epoch 1][Iter: 1/4]CELoss: 50.22282, loss: 50.22282, top1: 0.18750, top5: 1.00000, batch_cost: 17.80208s, reader_cost: 0.40996, ips: 7.19017 images/sec
    [2021/08/13 20:10:18] root INFO: [Eval][Epoch 1][Iter: 2/4]CELoss: 48.80469, loss: 48.80469, top1: 0.20312, top5: 1.00000, batch_cost: 17.66775s, reader_cost: 0.27335, ips: 7.24484 images/sec
    [2021/08/13 20:10:34] root INFO: [Eval][Epoch 1][Iter: 3/4]CELoss: 49.81577, loss: 49.81577, top1: 0.15517, top5: 1.00000, batch_cost: 17.16789s, reader_cost: 0.20505, ips: 6.75680 images/sec
    [2021/08/13 20:10:34] root INFO: [Eval][Epoch 1][Avg]CELoss: 49.46626, loss: 49.46626, top1: 0.18600, top5: 1.00000
    [2021/08/13 20:10:35] root INFO: Already save model in ./output/ResNet50/best_model
    [2021/08/13 20:10:35] root INFO: [Eval][Epoch 1][best metric: 0.1859999985694885]
    [2021/08/13 20:10:35] root INFO: Already save model in ./output/ResNet50/epoch_1
    [2021/08/13 20:10:36] root INFO: Already save model in ./output/ResNet50/latest
    [2021/08/13 20:11:32] root INFO: [Train][Epoch 2/2][Iter: 0/36]lr: 0.10000, CELoss: 2.20746, loss: 2.20746, top1: 0.22656, top5: 1.00000, batch_cost: 57.73352s, reader_cost: 2.21990, ips: 2.21708 images/sec, eta: 0:34:38
    [2021/08/13 20:12:28] root INFO: [Train][Epoch 2/2][Iter: 1/36]lr: 0.10000, CELoss: 1.99393, loss: 1.99393, top1: 0.19531, top5: 1.00000, batch_cost: 57.67393s, reader_cost: 2.15263, ips: 2.21937 images/sec, eta: 0:33:38
    [2021/08/13 20:13:23] root INFO: [Train][Epoch 2/2][Iter: 2/36]lr: 0.10000, CELoss: 2.05486, loss: 2.05486, top1: 0.17188, top5: 1.00000, batch_cost: 57.61328s, reader_cost: 2.08933, ips: 2.22171 images/sec, eta: 0:32:38
    [2021/08/13 20:14:19] root INFO: [Train][Epoch 2/2][Iter: 3/36]lr: 0.10000, CELoss: 2.06982, loss: 2.06982, top1: 0.17969, top5: 1.00000, batch_cost: 57.55112s, reader_cost: 2.02964, ips: 2.22411 images/sec, eta: 0:31:39
    [2021/08/13 20:15:14] root INFO: [Train][Epoch 2/2][Iter: 4/36]lr: 0.10000, CELoss: 2.11335, loss: 2.11335, top1: 0.18906, top5: 1.00000, batch_cost: 57.49249s, reader_cost: 1.97327, ips: 2.22638 images/sec, eta: 0:30:39
    [2021/08/13 20:16:10] root INFO: [Train][Epoch 2/2][Iter: 5/36]lr: 0.10000, CELoss: 2.09528, loss: 2.09528, top1: 0.18880, top5: 1.00000, batch_cost: 55.40142s, reader_cost: 0.00022, ips: 2.31041 images/sec, eta: 0:28:37
    [2021/08/13 20:17:05] root INFO: [Train][Epoch 2/2][Iter: 6/36]lr: 0.10000, CELoss: 2.05090, loss: 2.05090, top1: 0.19308, top5: 1.00000, batch_cost: 55.41494s, reader_cost: 0.00021, ips: 2.30985 images/sec, eta: 0:27:42
    [2021/08/13 20:18:01] root INFO: [Train][Epoch 2/2][Iter: 7/36]lr: 0.10000, CELoss: 2.06660, loss: 2.06660, top1: 0.18750, top5: 1.00000, batch_cost: 55.43053s, reader_cost: 0.00022, ips: 2.30920 images/sec, eta: 0:26:47
    [2021/08/13 20:18:56] root INFO: [Train][Epoch 2/2][Iter: 8/36]lr: 0.10000, CELoss: 2.07246, loss: 2.07246, top1: 0.18403, top5: 1.00000, batch_cost: 55.44267s, reader_cost: 0.00022, ips: 2.30869 images/sec, eta: 0:25:52
    [2021/08/13 20:19:51] root INFO: [Train][Epoch 2/2][Iter: 9/36]lr: 0.10000, CELoss: 2.03787, loss: 2.03787, top1: 0.18594, top5: 1.00000, batch_cost: 55.44202s, reader_cost: 0.00022, ips: 2.30872 images/sec, eta: 0:24:56
    [2021/08/13 20:20:47] root INFO: [Train][Epoch 2/2][Iter: 10/36]lr: 0.10000, CELoss: 2.01957, loss: 2.01957, top1: 0.18395, top5: 1.00000, batch_cost: 55.44990s, reader_cost: 0.00021, ips: 2.30839 images/sec, eta: 0:24:01
    [2021/08/13 20:21:42] root INFO: [Train][Epoch 2/2][Iter: 11/36]lr: 0.10000, CELoss: 1.98948, loss: 1.98948, top1: 0.18555, top5: 1.00000, batch_cost: 55.44889s, reader_cost: 0.00021, ips: 2.30843 images/sec, eta: 0:23:06
    [2021/08/13 20:22:38] root INFO: [Train][Epoch 2/2][Iter: 12/36]lr: 0.10000, CELoss: 1.96709, loss: 1.96709, top1: 0.18750, top5: 1.00000, batch_cost: 55.46224s, reader_cost: 0.00021, ips: 2.30788 images/sec, eta: 0:22:11
    [2021/08/13 20:23:33] root INFO: [Train][Epoch 2/2][Iter: 13/36]lr: 0.10000, CELoss: 1.95105, loss: 1.95105, top1: 0.18973, top5: 1.00000, batch_cost: 55.45261s, reader_cost: 0.00021, ips: 2.30828 images/sec, eta: 0:21:15
    [2021/08/13 20:24:29] root INFO: [Train][Epoch 2/2][Iter: 14/36]lr: 0.10000, CELoss: 1.96512, loss: 1.96512, top1: 0.18854, top5: 1.00000, batch_cost: 55.44777s, reader_cost: 0.00021, ips: 2.30848 images/sec, eta: 0:20:19
    [2021/08/13 20:25:24] root INFO: [Train][Epoch 2/2][Iter: 15/36]lr: 0.10000, CELoss: 1.94530, loss: 1.94530, top1: 0.18896, top5: 1.00000, batch_cost: 55.44317s, reader_cost: 0.00021, ips: 2.30867 images/sec, eta: 0:19:24
    [2021/08/13 20:26:19] root INFO: [Train][Epoch 2/2][Iter: 16/36]lr: 0.10000, CELoss: 1.92410, loss: 1.92410, top1: 0.18980, top5: 1.00000, batch_cost: 55.43688s, reader_cost: 0.00021, ips: 2.30893 images/sec, eta: 0:18:28
    [2021/08/13 20:27:15] root INFO: [Train][Epoch 2/2][Iter: 17/36]lr: 0.10000, CELoss: 1.90760, loss: 1.90760, top1: 0.19314, top5: 1.00000, batch_cost: 55.43422s, reader_cost: 0.00021, ips: 2.30904 images/sec, eta: 0:17:33
    [2021/08/13 20:28:10] root INFO: [Train][Epoch 2/2][Iter: 18/36]lr: 0.10000, CELoss: 1.91625, loss: 1.91625, top1: 0.19161, top5: 1.00000, batch_cost: 55.43241s, reader_cost: 0.00021, ips: 2.30912 images/sec, eta: 0:16:37
    [2021/08/13 20:29:06] root INFO: [Train][Epoch 2/2][Iter: 19/36]lr: 0.10000, CELoss: 1.90320, loss: 1.90320, top1: 0.19375, top5: 1.00000, batch_cost: 55.43059s, reader_cost: 0.00021, ips: 2.30919 images/sec, eta: 0:15:42
    [2021/08/13 20:30:01] root INFO: [Train][Epoch 2/2][Iter: 20/36]lr: 0.10000, CELoss: 1.90222, loss: 1.90222, top1: 0.19308, top5: 1.00000, batch_cost: 55.43161s, reader_cost: 0.00021, ips: 2.30915 images/sec, eta: 0:14:46
    [2021/08/13 20:30:57] root INFO: [Train][Epoch 2/2][Iter: 21/36]lr: 0.10000, CELoss: 1.89481, loss: 1.89481, top1: 0.19354, top5: 1.00000, batch_cost: 55.42955s, reader_cost: 0.00021, ips: 2.30924 images/sec, eta: 0:13:51
    [2021/08/13 20:31:52] root INFO: [Train][Epoch 2/2][Iter: 22/36]lr: 0.10000, CELoss: 1.88829, loss: 1.88829, top1: 0.19429, top5: 1.00000, batch_cost: 55.43464s, reader_cost: 0.00021, ips: 2.30903 images/sec, eta: 0:12:56
    [2021/08/13 20:32:48] root INFO: [Train][Epoch 2/2][Iter: 23/36]lr: 0.10000, CELoss: 1.89969, loss: 1.89969, top1: 0.19336, top5: 1.00000, batch_cost: 55.43414s, reader_cost: 0.00022, ips: 2.30905 images/sec, eta: 0:12:00
    [2021/08/13 20:33:43] root INFO: [Train][Epoch 2/2][Iter: 24/36]lr: 0.10000, CELoss: 1.88899, loss: 1.88899, top1: 0.19406, top5: 1.00000, batch_cost: 55.43132s, reader_cost: 0.00022, ips: 2.30916 images/sec, eta: 0:11:05
    [2021/08/13 20:34:38] root INFO: [Train][Epoch 2/2][Iter: 25/36]lr: 0.10000, CELoss: 1.87943, loss: 1.87943, top1: 0.19471, top5: 1.00000, batch_cost: 55.43108s, reader_cost: 0.00022, ips: 2.30917 images/sec, eta: 0:10:09
    [2021/08/13 20:35:34] root INFO: [Train][Epoch 2/2][Iter: 26/36]lr: 0.10000, CELoss: 1.87333, loss: 1.87333, top1: 0.19531, top5: 1.00000, batch_cost: 55.43283s, reader_cost: 0.00022, ips: 2.30910 images/sec, eta: 0:09:14
    [2021/08/13 20:36:29] root INFO: [Train][Epoch 2/2][Iter: 27/36]lr: 0.10000, CELoss: 1.86739, loss: 1.86739, top1: 0.19754, top5: 1.00000, batch_cost: 55.43460s, reader_cost: 0.00022, ips: 2.30903 images/sec, eta: 0:08:18
    [2021/08/13 20:37:25] root INFO: [Train][Epoch 2/2][Iter: 28/36]lr: 0.10000, CELoss: 1.86457, loss: 1.86457, top1: 0.19774, top5: 1.00000, batch_cost: 55.43636s, reader_cost: 0.00022, ips: 2.30895 images/sec, eta: 0:07:23
    [2021/08/13 20:38:20] root INFO: [Train][Epoch 2/2][Iter: 29/36]lr: 0.10000, CELoss: 1.85879, loss: 1.85879, top1: 0.19714, top5: 1.00000, batch_cost: 55.44015s, reader_cost: 0.00022, ips: 2.30880 images/sec, eta: 0:06:28
    [2021/08/13 20:39:16] root INFO: [Train][Epoch 2/2][Iter: 30/36]lr: 0.10000, CELoss: 1.85452, loss: 1.85452, top1: 0.19758, top5: 1.00000, batch_cost: 55.44296s, reader_cost: 0.00022, ips: 2.30868 images/sec, eta: 0:05:32
    [2021/08/13 20:40:11] root INFO: [Train][Epoch 2/2][Iter: 31/36]lr: 0.10000, CELoss: 1.84575, loss: 1.84575, top1: 0.19946, top5: 1.00000, batch_cost: 55.44466s, reader_cost: 0.00022, ips: 2.30861 images/sec, eta: 0:04:37
    [2021/08/13 20:41:07] root INFO: [Train][Epoch 2/2][Iter: 32/36]lr: 0.10000, CELoss: 1.84742, loss: 1.84742, top1: 0.20005, top5: 1.00000, batch_cost: 55.44561s, reader_cost: 0.00022, ips: 2.30857 images/sec, eta: 0:03:41
    [2021/08/13 20:42:02] root INFO: [Train][Epoch 2/2][Iter: 33/36]lr: 0.10000, CELoss: 1.84534, loss: 1.84534, top1: 0.20106, top5: 1.00000, batch_cost: 55.44332s, reader_cost: 0.00022, ips: 2.30866 images/sec, eta: 0:02:46
    [2021/08/13 20:42:57] root INFO: [Train][Epoch 2/2][Iter: 34/36]lr: 0.10000, CELoss: 1.83835, loss: 1.83835, top1: 0.20134, top5: 1.00000, batch_cost: 55.44093s, reader_cost: 0.00022, ips: 2.30876 images/sec, eta: 0:01:50
    [2021/08/13 20:43:06] root INFO: [Train][Epoch 2/2][Iter: 35/36]lr: 0.10000, CELoss: 1.83776, loss: 1.83776, top1: 0.20156, top5: 1.00000, batch_cost: 53.93681s, reader_cost: 0.00022, ips: 0.37080 images/sec, eta: 0:00:53
    [2021/08/13 20:43:06] root INFO: [Train][Epoch 2/2][Avg]CELoss: 1.83776, loss: 1.83776, top1: 0.20156, top5: 1.00000
    [2021/08/13 20:43:24] root INFO: [Eval][Epoch 2][Iter: 0/4]CELoss: 1.71736, loss: 1.71736, top1: 0.25781, top5: 1.00000, batch_cost: 18.15101s, reader_cost: 0.80468, ips: 7.05195 images/sec
    [2021/08/13 20:43:42] root INFO: [Eval][Epoch 2][Iter: 1/4]CELoss: 1.97634, loss: 1.97634, top1: 0.17188, top5: 1.00000, batch_cost: 17.74070s, reader_cost: 0.40241, ips: 7.21505 images/sec
    [2021/08/13 20:43:59] root INFO: [Eval][Epoch 2][Iter: 2/4]CELoss: 1.62544, loss: 1.62544, top1: 0.13281, top5: 1.00000, batch_cost: 17.60606s, reader_cost: 0.26832, ips: 7.27022 images/sec
    [2021/08/13 20:44:15] root INFO: [Eval][Epoch 2][Iter: 3/4]CELoss: 1.75792, loss: 1.75792, top1: 0.16379, top5: 1.00000, batch_cost: 17.13347s, reader_cost: 0.20128, ips: 6.77038 images/sec
    [2021/08/13 20:44:15] root INFO: [Eval][Epoch 2][Avg]CELoss: 1.76954, loss: 1.76954, top1: 0.18200, top5: 1.00000
    [2021/08/13 20:44:15] root INFO: [Eval][Epoch 2][best metric: 0.1859999985694885]
    [2021/08/13 20:44:15] root INFO: Already save model in ./output/ResNet50/epoch_2
    [2021/08/13 20:44:16] root INFO: Already save model in ./output/ResNet50/latest


### 3.7 预测一张


```python
# 更换为你训练的网络，需要预测的文件，上面训练所得到的的最优模型文件
# 我这里是不严谨的，直接使用训练集的图片进行验证，大家可以去百度搜一些相关的图片传上来，进行预测
!python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml \
    -o Infer.infer_imgs=dataset/foods/baby_back_ribs/319516.jpg \
    -o Global.pretrained_model=output/ResNet50/best_model
```

    /home/aistudio/PaddleClas/ppcls/arch/backbone/model_zoo/vision_transformer.py:15: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Callable
    [2021/08/13 20:56:06] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2021/08/13 20:56:06] root INFO: Arch : 
    [2021/08/13 20:56:06] root INFO:     class_num : 6
    [2021/08/13 20:56:06] root INFO:     name : ResNet50
    [2021/08/13 20:56:06] root INFO: DataLoader : 
    [2021/08/13 20:56:06] root INFO:     Eval : 
    [2021/08/13 20:56:06] root INFO:         dataset : 
    [2021/08/13 20:56:06] root INFO:             cls_label_path : ./dataset/foods/val_list.txt
    [2021/08/13 20:56:06] root INFO:             image_root : ./dataset/
    [2021/08/13 20:56:06] root INFO:             name : ImageNetDataset
    [2021/08/13 20:56:06] root INFO:             transform_ops : 
    [2021/08/13 20:56:06] root INFO:                 DecodeImage : 
    [2021/08/13 20:56:06] root INFO:                     channel_first : False
    [2021/08/13 20:56:06] root INFO:                     to_rgb : True
    [2021/08/13 20:56:06] root INFO:                 ResizeImage : 
    [2021/08/13 20:56:06] root INFO:                     resize_short : 256
    [2021/08/13 20:56:06] root INFO:                 CropImage : 
    [2021/08/13 20:56:06] root INFO:                     size : 224
    [2021/08/13 20:56:06] root INFO:                 NormalizeImage : 
    [2021/08/13 20:56:06] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/13 20:56:06] root INFO:                     order : 
    [2021/08/13 20:56:06] root INFO:                     scale : 1.0/255.0
    [2021/08/13 20:56:06] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/13 20:56:06] root INFO:         loader : 
    [2021/08/13 20:56:06] root INFO:             num_workers : 0
    [2021/08/13 20:56:06] root INFO:             use_shared_memory : True
    [2021/08/13 20:56:06] root INFO:         sampler : 
    [2021/08/13 20:56:06] root INFO:             batch_size : 128
    [2021/08/13 20:56:06] root INFO:             drop_last : False
    [2021/08/13 20:56:06] root INFO:             name : DistributedBatchSampler
    [2021/08/13 20:56:06] root INFO:             shuffle : True
    [2021/08/13 20:56:06] root INFO:     Train : 
    [2021/08/13 20:56:06] root INFO:         dataset : 
    [2021/08/13 20:56:06] root INFO:             cls_label_path : ./dataset/foods/train_list.txt
    [2021/08/13 20:56:06] root INFO:             image_root : ./dataset/
    [2021/08/13 20:56:06] root INFO:             name : ImageNetDataset
    [2021/08/13 20:56:06] root INFO:             transform_ops : 
    [2021/08/13 20:56:06] root INFO:                 DecodeImage : 
    [2021/08/13 20:56:06] root INFO:                     channel_first : False
    [2021/08/13 20:56:06] root INFO:                     to_rgb : True
    [2021/08/13 20:56:06] root INFO:                 ResizeImage : 
    [2021/08/13 20:56:06] root INFO:                     resize_short : 256
    [2021/08/13 20:56:06] root INFO:                 CropImage : 
    [2021/08/13 20:56:06] root INFO:                     size : 224
    [2021/08/13 20:56:06] root INFO:                 RandFlipImage : 
    [2021/08/13 20:56:06] root INFO:                     flip_code : 1
    [2021/08/13 20:56:06] root INFO:                 NormalizeImage : 
    [2021/08/13 20:56:06] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/13 20:56:06] root INFO:                     order : 
    [2021/08/13 20:56:06] root INFO:                     scale : 1.0/255.0
    [2021/08/13 20:56:06] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/13 20:56:06] root INFO:         loader : 
    [2021/08/13 20:56:06] root INFO:             num_workers : 0
    [2021/08/13 20:56:06] root INFO:             use_shared_memory : True
    [2021/08/13 20:56:06] root INFO:         sampler : 
    [2021/08/13 20:56:06] root INFO:             batch_size : 128
    [2021/08/13 20:56:06] root INFO:             drop_last : False
    [2021/08/13 20:56:06] root INFO:             name : DistributedBatchSampler
    [2021/08/13 20:56:06] root INFO:             shuffle : True
    [2021/08/13 20:56:06] root INFO: Global : 
    [2021/08/13 20:56:06] root INFO:     checkpoints : None
    [2021/08/13 20:56:06] root INFO:     device : cpu
    [2021/08/13 20:56:06] root INFO:     epochs : 2
    [2021/08/13 20:56:06] root INFO:     eval_during_train : True
    [2021/08/13 20:56:06] root INFO:     eval_interval : 1
    [2021/08/13 20:56:06] root INFO:     image_shape : [3, 224, 224]
    [2021/08/13 20:56:06] root INFO:     output_dir : ./output/
    [2021/08/13 20:56:06] root INFO:     pretrained_model : output/ResNet50/best_model
    [2021/08/13 20:56:06] root INFO:     print_batch_step : 1
    [2021/08/13 20:56:06] root INFO:     save_inference_dir : ./inference
    [2021/08/13 20:56:06] root INFO:     save_interval : 1
    [2021/08/13 20:56:06] root INFO:     to_static : False
    [2021/08/13 20:56:06] root INFO:     use_visualdl : True
    [2021/08/13 20:56:06] root INFO: Infer : 
    [2021/08/13 20:56:06] root INFO:     PostProcess : 
    [2021/08/13 20:56:06] root INFO:         class_id_map_file : ./dataset/label_list.txt
    [2021/08/13 20:56:06] root INFO:         name : Topk
    [2021/08/13 20:56:06] root INFO:         topk : 5
    [2021/08/13 20:56:06] root INFO:     batch_size : 10
    [2021/08/13 20:56:06] root INFO:     infer_imgs : dataset/foods/baby_back_ribs/319516.jpg
    [2021/08/13 20:56:06] root INFO:     transforms : 
    [2021/08/13 20:56:06] root INFO:         DecodeImage : 
    [2021/08/13 20:56:06] root INFO:             channel_first : False
    [2021/08/13 20:56:06] root INFO:             to_rgb : True
    [2021/08/13 20:56:06] root INFO:         ResizeImage : 
    [2021/08/13 20:56:06] root INFO:             resize_short : 256
    [2021/08/13 20:56:06] root INFO:         CropImage : 
    [2021/08/13 20:56:06] root INFO:             size : 224
    [2021/08/13 20:56:06] root INFO:         NormalizeImage : 
    [2021/08/13 20:56:06] root INFO:             mean : [0.485, 0.456, 0.406]
    [2021/08/13 20:56:06] root INFO:             order : 
    [2021/08/13 20:56:06] root INFO:             scale : 1.0/255.0
    [2021/08/13 20:56:06] root INFO:             std : [0.229, 0.224, 0.225]
    [2021/08/13 20:56:06] root INFO:         ToCHWImage : None
    [2021/08/13 20:56:06] root INFO: Loss : 
    [2021/08/13 20:56:06] root INFO:     Eval : 
    [2021/08/13 20:56:06] root INFO:         CELoss : 
    [2021/08/13 20:56:06] root INFO:             weight : 1.0
    [2021/08/13 20:56:06] root INFO:     Train : 
    [2021/08/13 20:56:06] root INFO:         CELoss : 
    [2021/08/13 20:56:06] root INFO:             weight : 1.0
    [2021/08/13 20:56:06] root INFO: Metric : 
    [2021/08/13 20:56:06] root INFO:     Eval : 
    [2021/08/13 20:56:06] root INFO:         TopkAcc : 
    [2021/08/13 20:56:06] root INFO:             topk : [1, 5]
    [2021/08/13 20:56:06] root INFO:     Train : 
    [2021/08/13 20:56:06] root INFO:         TopkAcc : 
    [2021/08/13 20:56:06] root INFO:             topk : [1, 5]
    [2021/08/13 20:56:06] root INFO: Optimizer : 
    [2021/08/13 20:56:06] root INFO:     lr : 
    [2021/08/13 20:56:06] root INFO:         decay_epochs : [30, 60, 90]
    [2021/08/13 20:56:06] root INFO:         learning_rate : 0.015
    [2021/08/13 20:56:06] root INFO:         name : Piecewise
    [2021/08/13 20:56:06] root INFO:         values : [0.1, 0.01, 0.001, 0.0001]
    [2021/08/13 20:56:06] root INFO:     momentum : 0.9
    [2021/08/13 20:56:06] root INFO:     name : Momentum
    [2021/08/13 20:56:06] root INFO:     regularizer : 
    [2021/08/13 20:56:06] root INFO:         coeff : 0.0005
    [2021/08/13 20:56:06] root INFO:         name : L2
    [2021/08/13 20:56:08] root INFO: train with paddle 2.1.2 and device CPUPlace
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    [{'class_ids': [2, 5, 4, 3, 1], 'scores': [1.0, 0.0, 0.0, 0.0, 0.0], 'file_name': 'dataset/foods/baby_back_ribs/319516.jpg', 'label_names': ['baby_back_ribs', 'baklava', 'apple_pie', 'beef_tartare', 'beef_carpaccio']}]


运行完成，最后几行会得到结果如下形式：
```
[{'class_ids': [5, 1, 3, 4, 2],
'scores': [0.48433, 0.26765, 0.13903, 0.05609, 0.05162],
'file_name': 'dataset/foods/baby_back_ribs/319516.jpg', 
'label_names': ['baklava', 'beef_carpaccio', 'beef_tartare', 'apple_pie', 'baby_back_ribs']}]
```
   
训练轮数还有很大提升空间，自行变动参数直到预测正确为止~    



# 四、效果展示
```
[{'class_ids': [2, 5, 4, 3, 1], 
'scores': [1.0, 0.0, 0.0, 0.0, 0.0], 
'file_name': 'dataset/foods/baby_back_ribs/319516.jpg', 
'label_names': ['baby_back_ribs', 'baklava', 'apple_pie', 'beef_tartare', 'beef_carpaccio']}]
```

# 五、总结与升华


非常感谢百度公司提供了这次学习活动，让我知道了如何进行初步的，完成一些项目

这次作业中也遇到了不少困难

但因为有老师和同学们的帮助最终克服了它

在修改ShuffleNetV2_x0_25.yaml文件中遇到了一些困难



# 个人简介

我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/538518
