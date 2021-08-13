# [AI训练营]paddleclas实现图像分类baseline

## 0 项目背景
> [课程链接：飞桨领航团AI达人创造营](https://aistudio.baidu.com/aistudio/course/introduce/24607?directly=1&shared=1)    

<hr>
<center><font size="3px" color="green">暑    期    充    电    季</font></center>
<hr>
<center><font size="2px" color="red">百度飞桨领航团全新推出“AI达人创造营”</font></center>
<center><font size="2px" color="green"><strong>十位飞桨开发者技术专家（PPDE）</strong>手把手教大家完成项目从idea思考到部署落地的全流程实战</font></center>
<center><font >最终让每位参与者都有一个可以给自己简历加分的项目</font></center>
<center><font >7月26日-8月16日，每晚 19:00-21:00 直播讲解、十位飞桨开发者技术专家（PPDE）手把手教助你成为AI达人</font></center>   
<center><font size="2px" color="blue">报名后，请加入课程 QQ 群861942585，QQ群用于直播提醒、实时答疑、交流互动等</font></center>  

<hr>
<font size="2px" color="red">PS:</font>

<font size="3px" color="black">本项目属于本次课程的大作业的一部分，希望大家可以学会使用paddleclas实现图像分类。</font>
<hr>

## 本大作业任务

1、选择一个心仪的数据集     
2、运行项目，<font color="red">能够跑通项目即可达到结业要求</font>     
3、记得生成新版本，公开项目哦~

**加分项：**     
4、更换网络、进行数据预处理和调参   

## 1 项目实现简介
正如标题，采用paddleclas套件实现分类[30分钟玩转PaddleClas（尝鲜版）](https://gitee.com/paddlepaddle/PaddleClas/blob/release/2.2/docs/zh_CN/tutorials/quick_start_new_user.md)    
查看套件，可以知道    
实现分类，仅仅需要我们将数据集提取为如下这种格式的txt文件即可（当然我们并不需要更大的训练集）    
![](https://ai-studio-static-online.cdn.bcebos.com/012e5e2e7a204f77ab2ad9759860978d6b19d55946dd4ee2adea6a4e9736d121)

PS：
如有需要参考项目可看：      
[基于PaddleClas2.2的从零到落地安卓部署的奥特曼分类实战](https://aistudio.baidu.com/aistudio/projectdetail/2219455)
[iFLYTEK基于PaddleClas2.2的广告分类baseline非官方](https://aistudio.baidu.com/aistudio/projectdetail/2235340)


## 2 数据集介绍

<font size="3px" color="red">本次数据集有五个可供大家选择。分别是：</font>   
1. 猫12分类
2. 垃圾40分类
3. 场景5分类
4. 食物5分类
5. 蝴蝶20分类

**数据集：都是不同类别的文件夹下放置了对应文件夹名字的类别图片**



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

### 2.1 解压数据集，查看数据的结构


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

### 2.2 拿到总的训练数据txt


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


### 2.3 数据打乱


```python
# 把数据打乱
all_list = shuffle(all_list)
allstr = ''.join(all_list)
f = open('all_list.txt','w',encoding='utf-8')
f.write(allstr)
print("打乱成功，并重新写入文本")
```

    打乱成功，并重新写入文本


### 2.4 数据划分


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


## 3 安装paddleclas

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
    Receiving objects: 100% (15290/15290), 113.56 MiB | 13.02 MiB/s, done.
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

### 3.1 修改配置文件
#### 3.1.1
主要是以下几点：分类数、图片总量、训练和验证的路径、图像尺寸、数据预处理、训练和预测的num_workers: 0

路径如下：
>PaddleClas/ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml

<font size="3px" color="red">（主要的参数已经进行注释，一定要过一遍）</font>

```
# global configs
Global:
  checkpoints: null
  pretrained_model: null
  output_dir: ./output/
  # 使用GPU训练
  device: gpu
  # 每几个轮次保存一次
  save_interval: 1 
  eval_during_train: True
  # 每几个轮次验证一次
  eval_interval: 1 
  # 训练轮次
  epochs: 20 
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
#### 3.1.2 标签文件
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

### 3.2 开始训练


```python
# 提示，运行过程中可能存在坏图的情况，但是不用担心，训练过程不受影响。
# 仅供参考，我只跑了五轮，准确率很低
!python3 tools/train.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml
```

    /home/aistudio/PaddleClas/ppcls/arch/backbone/model_zoo/vision_transformer.py:15: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Callable
    [2021/08/13 10:42:46] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2021/08/13 10:42:46] root INFO: Arch : 
    [2021/08/13 10:42:46] root INFO:     class_num : 6
    [2021/08/13 10:42:46] root INFO:     name : ResNet50
    [2021/08/13 10:42:46] root INFO: DataLoader : 
    [2021/08/13 10:42:46] root INFO:     Eval : 
    [2021/08/13 10:42:46] root INFO:         dataset : 
    [2021/08/13 10:42:46] root INFO:             cls_label_path : ./dataset/foods/val_list.txt
    [2021/08/13 10:42:46] root INFO:             image_root : ./dataset/
    [2021/08/13 10:42:46] root INFO:             name : ImageNetDataset
    [2021/08/13 10:42:46] root INFO:             transform_ops : 
    [2021/08/13 10:42:46] root INFO:                 DecodeImage : 
    [2021/08/13 10:42:46] root INFO:                     channel_first : False
    [2021/08/13 10:42:46] root INFO:                     to_rgb : True
    [2021/08/13 10:42:46] root INFO:                 ResizeImage : 
    [2021/08/13 10:42:46] root INFO:                     resize_short : 256
    [2021/08/13 10:42:46] root INFO:                 CropImage : 
    [2021/08/13 10:42:46] root INFO:                     size : 224
    [2021/08/13 10:42:46] root INFO:                 NormalizeImage : 
    [2021/08/13 10:42:46] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/13 10:42:46] root INFO:                     order : 
    [2021/08/13 10:42:46] root INFO:                     scale : 1.0/255.0
    [2021/08/13 10:42:46] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/13 10:42:46] root INFO:         loader : 
    [2021/08/13 10:42:46] root INFO:             num_workers : 0
    [2021/08/13 10:42:46] root INFO:             use_shared_memory : True
    [2021/08/13 10:42:46] root INFO:         sampler : 
    [2021/08/13 10:42:46] root INFO:             batch_size : 128
    [2021/08/13 10:42:46] root INFO:             drop_last : False
    [2021/08/13 10:42:46] root INFO:             name : DistributedBatchSampler
    [2021/08/13 10:42:46] root INFO:             shuffle : True
    [2021/08/13 10:42:46] root INFO:     Train : 
    [2021/08/13 10:42:46] root INFO:         dataset : 
    [2021/08/13 10:42:46] root INFO:             cls_label_path : ./dataset/foods/train_list.txt
    [2021/08/13 10:42:46] root INFO:             image_root : ./dataset/
    [2021/08/13 10:42:46] root INFO:             name : ImageNetDataset
    [2021/08/13 10:42:46] root INFO:             transform_ops : 
    [2021/08/13 10:42:46] root INFO:                 DecodeImage : 
    [2021/08/13 10:42:46] root INFO:                     channel_first : False
    [2021/08/13 10:42:46] root INFO:                     to_rgb : True
    [2021/08/13 10:42:46] root INFO:                 ResizeImage : 
    [2021/08/13 10:42:46] root INFO:                     resize_short : 256
    [2021/08/13 10:42:46] root INFO:                 CropImage : 
    [2021/08/13 10:42:46] root INFO:                     size : 224
    [2021/08/13 10:42:46] root INFO:                 RandFlipImage : 
    [2021/08/13 10:42:46] root INFO:                     flip_code : 1
    [2021/08/13 10:42:46] root INFO:                 NormalizeImage : 
    [2021/08/13 10:42:46] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/13 10:42:46] root INFO:                     order : 
    [2021/08/13 10:42:46] root INFO:                     scale : 1.0/255.0
    [2021/08/13 10:42:46] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/13 10:42:46] root INFO:         loader : 
    [2021/08/13 10:42:46] root INFO:             num_workers : 0
    [2021/08/13 10:42:46] root INFO:             use_shared_memory : True
    [2021/08/13 10:42:46] root INFO:         sampler : 
    [2021/08/13 10:42:46] root INFO:             batch_size : 128
    [2021/08/13 10:42:46] root INFO:             drop_last : False
    [2021/08/13 10:42:46] root INFO:             name : DistributedBatchSampler
    [2021/08/13 10:42:46] root INFO:             shuffle : True
    [2021/08/13 10:42:46] root INFO: Global : 
    [2021/08/13 10:42:46] root INFO:     checkpoints : None
    [2021/08/13 10:42:46] root INFO:     device : cpu
    [2021/08/13 10:42:46] root INFO:     epochs : 20
    [2021/08/13 10:42:46] root INFO:     eval_during_train : True
    [2021/08/13 10:42:46] root INFO:     eval_interval : 1
    [2021/08/13 10:42:46] root INFO:     image_shape : [3, 224, 224]
    [2021/08/13 10:42:46] root INFO:     output_dir : ./output/
    [2021/08/13 10:42:46] root INFO:     pretrained_model : None
    [2021/08/13 10:42:46] root INFO:     print_batch_step : 1
    [2021/08/13 10:42:46] root INFO:     save_inference_dir : ./inference
    [2021/08/13 10:42:46] root INFO:     save_interval : 1
    [2021/08/13 10:42:46] root INFO:     to_static : False
    [2021/08/13 10:42:46] root INFO:     use_visualdl : True
    [2021/08/13 10:42:46] root INFO: Infer : 
    [2021/08/13 10:42:46] root INFO:     PostProcess : 
    [2021/08/13 10:42:46] root INFO:         class_id_map_file : ./dataset/label_list.txt
    [2021/08/13 10:42:46] root INFO:         name : Topk
    [2021/08/13 10:42:46] root INFO:         topk : 5
    [2021/08/13 10:42:46] root INFO:     batch_size : 10
    [2021/08/13 10:42:46] root INFO:     infer_imgs : ./dataset/foods/beef_carpaccio/855780.jpg
    [2021/08/13 10:42:46] root INFO:     transforms : 
    [2021/08/13 10:42:46] root INFO:         DecodeImage : 
    [2021/08/13 10:42:46] root INFO:             channel_first : False
    [2021/08/13 10:42:46] root INFO:             to_rgb : True
    [2021/08/13 10:42:46] root INFO:         ResizeImage : 
    [2021/08/13 10:42:46] root INFO:             resize_short : 256
    [2021/08/13 10:42:46] root INFO:         CropImage : 
    [2021/08/13 10:42:46] root INFO:             size : 224
    [2021/08/13 10:42:46] root INFO:         NormalizeImage : 
    [2021/08/13 10:42:46] root INFO:             mean : [0.485, 0.456, 0.406]
    [2021/08/13 10:42:46] root INFO:             order : 
    [2021/08/13 10:42:46] root INFO:             scale : 1.0/255.0
    [2021/08/13 10:42:46] root INFO:             std : [0.229, 0.224, 0.225]
    [2021/08/13 10:42:46] root INFO:         ToCHWImage : None
    [2021/08/13 10:42:46] root INFO: Loss : 
    [2021/08/13 10:42:46] root INFO:     Eval : 
    [2021/08/13 10:42:46] root INFO:         CELoss : 
    [2021/08/13 10:42:46] root INFO:             weight : 1.0
    [2021/08/13 10:42:46] root INFO:     Train : 
    [2021/08/13 10:42:46] root INFO:         CELoss : 
    [2021/08/13 10:42:46] root INFO:             weight : 1.0
    [2021/08/13 10:42:46] root INFO: Metric : 
    [2021/08/13 10:42:46] root INFO:     Eval : 
    [2021/08/13 10:42:46] root INFO:         TopkAcc : 
    [2021/08/13 10:42:46] root INFO:             topk : [1, 5]
    [2021/08/13 10:42:46] root INFO:     Train : 
    [2021/08/13 10:42:46] root INFO:         TopkAcc : 
    [2021/08/13 10:42:46] root INFO:             topk : [1, 5]
    [2021/08/13 10:42:46] root INFO: Optimizer : 
    [2021/08/13 10:42:46] root INFO:     lr : 
    [2021/08/13 10:42:46] root INFO:         decay_epochs : [30, 60, 90]
    [2021/08/13 10:42:46] root INFO:         learning_rate : 0.015
    [2021/08/13 10:42:46] root INFO:         name : Piecewise
    [2021/08/13 10:42:46] root INFO:         values : [0.1, 0.01, 0.001, 0.0001]
    [2021/08/13 10:42:46] root INFO:     momentum : 0.9
    [2021/08/13 10:42:46] root INFO:     name : Momentum
    [2021/08/13 10:42:46] root INFO:     regularizer : 
    [2021/08/13 10:42:46] root INFO:         coeff : 0.0005
    [2021/08/13 10:42:46] root INFO:         name : L2
    [2021/08/13 10:42:47] root INFO: train with paddle 2.1.2 and device CPUPlace
    {'CELoss': {'weight': 1.0}}
    [2021/08/13 10:43:49] root INFO: [Train][Epoch 1/20][Iter: 0/36]lr: 0.10000, CELoss: 2.02106, loss: 2.02106, top1: 0.07031, top5: 0.75781, batch_cost: 61.49253s, reader_cost: 0.90255, ips: 2.08155 images/sec, eta: 12:17:54
    [2021/08/13 10:44:45] root INFO: [Train][Epoch 1/20][Iter: 1/36]lr: 0.10000, CELoss: 8.09870, loss: 8.09870, top1: 0.13672, top5: 0.87891, batch_cost: 58.89962s, reader_cost: 0.45138, ips: 2.17319 images/sec, eta: 11:45:48
    [2021/08/13 10:45:41] root INFO: [Train][Epoch 1/20][Iter: 2/36]lr: 0.10000, CELoss: 16.09293, loss: 16.09293, top1: 0.15625, top5: 0.83854, batch_cost: 57.97555s, reader_cost: 0.30099, ips: 2.20783 images/sec, eta: 11:33:46
    [2021/08/13 10:46:37] root INFO: [Train][Epoch 1/20][Iter: 3/36]lr: 0.10000, CELoss: 17.55618, loss: 17.55618, top1: 0.16797, top5: 0.82227, batch_cost: 57.51679s, reader_cost: 0.22579, ips: 2.22544 images/sec, eta: 11:27:19
    [2021/08/13 10:47:33] root INFO: [Train][Epoch 1/20][Iter: 4/36]lr: 0.10000, CELoss: 17.29653, loss: 17.29653, top1: 0.16562, top5: 0.81406, batch_cost: 57.23357s, reader_cost: 0.18067, ips: 2.23645 images/sec, eta: 11:22:59
    [2021/08/13 10:48:29] root INFO: [Train][Epoch 1/20][Iter: 5/36]lr: 0.10000, CELoss: 17.08710, loss: 17.08710, top1: 0.16536, top5: 0.80990, batch_cost: 56.02482s, reader_cost: 0.00022, ips: 2.28470 images/sec, eta: 11:07:37
    [2021/08/13 10:49:25] root INFO: [Train][Epoch 1/20][Iter: 6/36]lr: 0.10000, CELoss: 17.74428, loss: 17.74428, top1: 0.17076, top5: 0.80804, batch_cost: 56.04137s, reader_cost: 0.00021, ips: 2.28403 images/sec, eta: 11:06:53
    [2021/08/13 10:50:21] root INFO: [Train][Epoch 1/20][Iter: 7/36]lr: 0.10000, CELoss: 16.11109, loss: 16.11109, top1: 0.17578, top5: 0.83203, batch_cost: 56.03565s, reader_cost: 0.00021, ips: 2.28426 images/sec, eta: 11:05:53
    [2021/08/13 10:51:17] root INFO: [Train][Epoch 1/20][Iter: 8/36]lr: 0.10000, CELoss: 14.67869, loss: 14.67869, top1: 0.18490, top5: 0.85069, batch_cost: 56.03431s, reader_cost: 0.00021, ips: 2.28431 images/sec, eta: 11:04:56
    [2021/08/13 10:52:13] root INFO: [Train][Epoch 1/20][Iter: 9/36]lr: 0.10000, CELoss: 13.54414, loss: 13.54414, top1: 0.18906, top5: 0.86562, batch_cost: 56.04303s, reader_cost: 0.00021, ips: 2.28396 images/sec, eta: 11:04:06
    [2021/08/13 10:53:09] root INFO: [Train][Epoch 1/20][Iter: 10/36]lr: 0.10000, CELoss: 13.06223, loss: 13.06223, top1: 0.19247, top5: 0.87784, batch_cost: 56.03309s, reader_cost: 0.00021, ips: 2.28436 images/sec, eta: 11:03:03
    [2021/08/13 10:54:06] root INFO: [Train][Epoch 1/20][Iter: 11/36]lr: 0.10000, CELoss: 12.38753, loss: 12.38753, top1: 0.19596, top5: 0.88802, batch_cost: 56.04862s, reader_cost: 0.00020, ips: 2.28373 images/sec, eta: 11:02:18
    [2021/08/13 10:55:02] root INFO: [Train][Epoch 1/20][Iter: 12/36]lr: 0.10000, CELoss: 11.74727, loss: 11.74727, top1: 0.19231, top5: 0.89663, batch_cost: 56.04846s, reader_cost: 0.00020, ips: 2.28374 images/sec, eta: 11:01:22
    [2021/08/13 10:55:58] root INFO: [Train][Epoch 1/20][Iter: 13/36]lr: 0.10000, CELoss: 11.15921, loss: 11.15921, top1: 0.19029, top5: 0.90402, batch_cost: 56.04971s, reader_cost: 0.00020, ips: 2.28369 images/sec, eta: 11:00:27
    [2021/08/13 10:56:54] root INFO: [Train][Epoch 1/20][Iter: 14/36]lr: 0.10000, CELoss: 10.66250, loss: 10.66250, top1: 0.19115, top5: 0.91042, batch_cost: 56.05066s, reader_cost: 0.00020, ips: 2.28365 images/sec, eta: 10:59:31
    [2021/08/13 10:57:50] root INFO: [Train][Epoch 1/20][Iter: 15/36]lr: 0.10000, CELoss: 10.23945, loss: 10.23945, top1: 0.18945, top5: 0.91602, batch_cost: 56.04858s, reader_cost: 0.00020, ips: 2.28373 images/sec, eta: 10:58:34
    [2021/08/13 10:58:46] root INFO: [Train][Epoch 1/20][Iter: 16/36]lr: 0.10000, CELoss: 9.81209, loss: 9.81209, top1: 0.18980, top5: 0.92096, batch_cost: 56.04942s, reader_cost: 0.00020, ips: 2.28370 images/sec, eta: 10:57:38
    [2021/08/13 10:59:42] root INFO: [Train][Epoch 1/20][Iter: 17/36]lr: 0.10000, CELoss: 9.42419, loss: 9.42419, top1: 0.19141, top5: 0.92535, batch_cost: 56.05368s, reader_cost: 0.00020, ips: 2.28353 images/sec, eta: 10:56:45
    [2021/08/13 11:00:38] root INFO: [Train][Epoch 1/20][Iter: 18/36]lr: 0.10000, CELoss: 9.13954, loss: 9.13954, top1: 0.19161, top5: 0.92928, batch_cost: 56.05591s, reader_cost: 0.00020, ips: 2.28343 images/sec, eta: 10:55:51
    [2021/08/13 11:01:34] root INFO: [Train][Epoch 1/20][Iter: 19/36]lr: 0.10000, CELoss: 8.81815, loss: 8.81815, top1: 0.19141, top5: 0.93281, batch_cost: 56.05334s, reader_cost: 0.00020, ips: 2.28354 images/sec, eta: 10:54:53
    [2021/08/13 11:02:30] root INFO: [Train][Epoch 1/20][Iter: 20/36]lr: 0.10000, CELoss: 8.50438, loss: 8.50438, top1: 0.19382, top5: 0.93601, batch_cost: 56.05995s, reader_cost: 0.00020, ips: 2.28327 images/sec, eta: 10:54:01
    [2021/08/13 11:03:26] root INFO: [Train][Epoch 1/20][Iter: 21/36]lr: 0.10000, CELoss: 8.21539, loss: 8.21539, top1: 0.19354, top5: 0.93892, batch_cost: 56.05699s, reader_cost: 0.00020, ips: 2.28339 images/sec, eta: 10:53:03
    [2021/08/13 11:04:22] root INFO: [Train][Epoch 1/20][Iter: 22/36]lr: 0.10000, CELoss: 7.93792, loss: 7.93792, top1: 0.19395, top5: 0.94158, batch_cost: 56.05516s, reader_cost: 0.00020, ips: 2.28346 images/sec, eta: 10:52:06
    [2021/08/13 11:05:18] root INFO: [Train][Epoch 1/20][Iter: 23/36]lr: 0.10000, CELoss: 7.69980, loss: 7.69980, top1: 0.19076, top5: 0.94401, batch_cost: 56.05099s, reader_cost: 0.00020, ips: 2.28363 images/sec, eta: 10:51:07
    [2021/08/13 11:06:14] root INFO: [Train][Epoch 1/20][Iter: 24/36]lr: 0.10000, CELoss: 7.47239, loss: 7.47239, top1: 0.18969, top5: 0.94625, batch_cost: 56.05377s, reader_cost: 0.00020, ips: 2.28352 images/sec, eta: 10:50:13
    [2021/08/13 11:07:10] root INFO: [Train][Epoch 1/20][Iter: 25/36]lr: 0.10000, CELoss: 7.27829, loss: 7.27829, top1: 0.19141, top5: 0.94832, batch_cost: 56.05519s, reader_cost: 0.00020, ips: 2.28346 images/sec, eta: 10:49:18
    [2021/08/13 11:08:07] root INFO: [Train][Epoch 1/20][Iter: 26/36]lr: 0.10000, CELoss: 7.06692, loss: 7.06692, top1: 0.19271, top5: 0.95023, batch_cost: 56.05827s, reader_cost: 0.00020, ips: 2.28334 images/sec, eta: 10:48:24
    [2021/08/13 11:09:03] root INFO: [Train][Epoch 1/20][Iter: 27/36]lr: 0.10000, CELoss: 6.87838, loss: 6.87838, top1: 0.19531, top5: 0.95201, batch_cost: 56.05607s, reader_cost: 0.00021, ips: 2.28343 images/sec, eta: 10:47:26
    [2021/08/13 11:09:59] root INFO: [Train][Epoch 1/20][Iter: 28/36]lr: 0.10000, CELoss: 6.71335, loss: 6.71335, top1: 0.19693, top5: 0.95366, batch_cost: 56.05846s, reader_cost: 0.00021, ips: 2.28333 images/sec, eta: 10:46:32
    [2021/08/13 11:10:55] root INFO: [Train][Epoch 1/20][Iter: 29/36]lr: 0.10000, CELoss: 6.55628, loss: 6.55628, top1: 0.19714, top5: 0.95521, batch_cost: 56.05635s, reader_cost: 0.00021, ips: 2.28342 images/sec, eta: 10:45:34
    [2021/08/13 11:11:51] root INFO: [Train][Epoch 1/20][Iter: 30/36]lr: 0.10000, CELoss: 6.41123, loss: 6.41123, top1: 0.19708, top5: 0.95665, batch_cost: 56.05384s, reader_cost: 0.00021, ips: 2.28352 images/sec, eta: 10:44:37
    [2021/08/13 11:12:47] root INFO: [Train][Epoch 1/20][Iter: 31/36]lr: 0.10000, CELoss: 6.27318, loss: 6.27318, top1: 0.19849, top5: 0.95801, batch_cost: 56.05332s, reader_cost: 0.00021, ips: 2.28354 images/sec, eta: 10:43:40
    [2021/08/13 11:13:43] root INFO: [Train][Epoch 1/20][Iter: 32/36]lr: 0.10000, CELoss: 6.15251, loss: 6.15251, top1: 0.19768, top5: 0.95928, batch_cost: 56.05122s, reader_cost: 0.00021, ips: 2.28363 images/sec, eta: 10:42:43
    [2021/08/13 11:14:39] root INFO: [Train][Epoch 1/20][Iter: 33/36]lr: 0.10000, CELoss: 6.02122, loss: 6.02122, top1: 0.19876, top5: 0.96048, batch_cost: 56.04857s, reader_cost: 0.00046, ips: 2.28373 images/sec, eta: 10:41:45
    [2021/08/13 11:15:35] root INFO: [Train][Epoch 1/20][Iter: 34/36]lr: 0.10000, CELoss: 5.90154, loss: 5.90154, top1: 0.19844, top5: 0.96161, batch_cost: 56.04792s, reader_cost: 0.00045, ips: 2.28376 images/sec, eta: 10:40:48
    [2021/08/13 11:15:44] root INFO: [Train][Epoch 1/20][Iter: 35/36]lr: 0.10000, CELoss: 5.88631, loss: 5.88631, top1: 0.19867, top5: 0.96178, batch_cost: 54.52781s, reader_cost: 0.00045, ips: 0.36679 images/sec, eta: 10:22:31
    [2021/08/13 11:15:44] root INFO: [Train][Epoch 1/20][Avg]CELoss: 5.88631, loss: 5.88631, top1: 0.19867, top5: 0.96178
    {'CELoss': {'weight': 1.0}}
    [2021/08/13 11:16:02] root INFO: [Eval][Epoch 1][Iter: 0/4]CELoss: 51.05238, loss: 51.05238, top1: 0.18750, top5: 1.00000, batch_cost: 18.31634s, reader_cost: 0.85357, ips: 6.98830 images/sec
    [2021/08/13 11:16:19] root INFO: [Eval][Epoch 1][Iter: 1/4]CELoss: 51.75068, loss: 51.75068, top1: 0.18750, top5: 1.00000, batch_cost: 17.86967s, reader_cost: 0.42685, ips: 7.16298 images/sec
    [2021/08/13 11:16:37] root INFO: [Eval][Epoch 1][Iter: 2/4]CELoss: 48.76045, loss: 48.76045, top1: 0.22656, top5: 1.00000, batch_cost: 17.73897s, reader_cost: 0.28461, ips: 7.21575 images/sec
    [2021/08/13 11:16:53] root INFO: [Eval][Epoch 1][Iter: 3/4]CELoss: 46.45121, loss: 46.45121, top1: 0.26724, top5: 1.00000, batch_cost: 17.26107s, reader_cost: 0.21349, ips: 6.72032 images/sec
    [2021/08/13 11:16:53] root INFO: [Eval][Epoch 1][Avg]CELoss: 49.57694, loss: 49.57694, top1: 0.21600, top5: 1.00000
    [2021/08/13 11:16:53] root INFO: Already save model in ./output/ResNet50/best_model
    [2021/08/13 11:16:53] root INFO: [Eval][Epoch 1][best metric: 0.2160000021457672]
    [2021/08/13 11:16:54] root INFO: Already save model in ./output/ResNet50/epoch_1
    [2021/08/13 11:16:54] root INFO: Already save model in ./output/ResNet50/latest
    [2021/08/13 11:17:51] root INFO: [Train][Epoch 2/20][Iter: 0/36]lr: 0.10000, CELoss: 1.64523, loss: 1.64523, top1: 0.23438, top5: 1.00000, batch_cost: 56.80453s, reader_cost: 2.22910, ips: 2.25334 images/sec, eta: 10:47:34
    [2021/08/13 11:18:47] root INFO: [Train][Epoch 2/20][Iter: 1/36]lr: 0.10000, CELoss: 1.88675, loss: 1.88675, top1: 0.20312, top5: 1.00000, batch_cost: 56.78248s, reader_cost: 2.16156, ips: 2.25422 images/sec, eta: 10:46:22
    [2021/08/13 11:19:43] root INFO: [Train][Epoch 2/20][Iter: 2/36]lr: 0.10000, CELoss: 1.83209, loss: 1.83209, top1: 0.20573, top5: 1.00000, batch_cost: 56.76282s, reader_cost: 2.09799, ips: 2.25500 images/sec, eta: 10:45:12
    [2021/08/13 11:20:39] root INFO: [Train][Epoch 2/20][Iter: 3/36]lr: 0.10000, CELoss: 1.91691, loss: 1.91691, top1: 0.20117, top5: 1.00000, batch_cost: 56.73985s, reader_cost: 2.03805, ips: 2.25591 images/sec, eta: 10:43:59
    [2021/08/13 11:21:35] root INFO: [Train][Epoch 2/20][Iter: 4/36]lr: 0.10000, CELoss: 1.92801, loss: 1.92801, top1: 0.19531, top5: 1.00000, batch_cost: 56.71708s, reader_cost: 1.98145, ips: 2.25682 images/sec, eta: 10:42:47
    ^C


### 3.3 预测一张


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
    [2021/08/13 11:22:15] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2021/08/13 11:22:15] root INFO: Arch : 
    [2021/08/13 11:22:15] root INFO:     class_num : 6
    [2021/08/13 11:22:15] root INFO:     name : ResNet50
    [2021/08/13 11:22:15] root INFO: DataLoader : 
    [2021/08/13 11:22:15] root INFO:     Eval : 
    [2021/08/13 11:22:15] root INFO:         dataset : 
    [2021/08/13 11:22:15] root INFO:             cls_label_path : ./dataset/foods/val_list.txt
    [2021/08/13 11:22:15] root INFO:             image_root : ./dataset/
    [2021/08/13 11:22:15] root INFO:             name : ImageNetDataset
    [2021/08/13 11:22:15] root INFO:             transform_ops : 
    [2021/08/13 11:22:15] root INFO:                 DecodeImage : 
    [2021/08/13 11:22:15] root INFO:                     channel_first : False
    [2021/08/13 11:22:15] root INFO:                     to_rgb : True
    [2021/08/13 11:22:15] root INFO:                 ResizeImage : 
    [2021/08/13 11:22:15] root INFO:                     resize_short : 256
    [2021/08/13 11:22:15] root INFO:                 CropImage : 
    [2021/08/13 11:22:15] root INFO:                     size : 224
    [2021/08/13 11:22:15] root INFO:                 NormalizeImage : 
    [2021/08/13 11:22:15] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/13 11:22:15] root INFO:                     order : 
    [2021/08/13 11:22:15] root INFO:                     scale : 1.0/255.0
    [2021/08/13 11:22:15] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/13 11:22:15] root INFO:         loader : 
    [2021/08/13 11:22:15] root INFO:             num_workers : 0
    [2021/08/13 11:22:15] root INFO:             use_shared_memory : True
    [2021/08/13 11:22:15] root INFO:         sampler : 
    [2021/08/13 11:22:15] root INFO:             batch_size : 128
    [2021/08/13 11:22:15] root INFO:             drop_last : False
    [2021/08/13 11:22:15] root INFO:             name : DistributedBatchSampler
    [2021/08/13 11:22:15] root INFO:             shuffle : True
    [2021/08/13 11:22:15] root INFO:     Train : 
    [2021/08/13 11:22:15] root INFO:         dataset : 
    [2021/08/13 11:22:15] root INFO:             cls_label_path : ./dataset/foods/train_list.txt
    [2021/08/13 11:22:15] root INFO:             image_root : ./dataset/
    [2021/08/13 11:22:15] root INFO:             name : ImageNetDataset
    [2021/08/13 11:22:15] root INFO:             transform_ops : 
    [2021/08/13 11:22:15] root INFO:                 DecodeImage : 
    [2021/08/13 11:22:15] root INFO:                     channel_first : False
    [2021/08/13 11:22:15] root INFO:                     to_rgb : True
    [2021/08/13 11:22:15] root INFO:                 ResizeImage : 
    [2021/08/13 11:22:15] root INFO:                     resize_short : 256
    [2021/08/13 11:22:15] root INFO:                 CropImage : 
    [2021/08/13 11:22:15] root INFO:                     size : 224
    [2021/08/13 11:22:15] root INFO:                 RandFlipImage : 
    [2021/08/13 11:22:15] root INFO:                     flip_code : 1
    [2021/08/13 11:22:15] root INFO:                 NormalizeImage : 
    [2021/08/13 11:22:15] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/13 11:22:15] root INFO:                     order : 
    [2021/08/13 11:22:15] root INFO:                     scale : 1.0/255.0
    [2021/08/13 11:22:15] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/13 11:22:15] root INFO:         loader : 
    [2021/08/13 11:22:15] root INFO:             num_workers : 0
    [2021/08/13 11:22:15] root INFO:             use_shared_memory : True
    [2021/08/13 11:22:15] root INFO:         sampler : 
    [2021/08/13 11:22:15] root INFO:             batch_size : 128
    [2021/08/13 11:22:15] root INFO:             drop_last : False
    [2021/08/13 11:22:15] root INFO:             name : DistributedBatchSampler
    [2021/08/13 11:22:15] root INFO:             shuffle : True
    [2021/08/13 11:22:15] root INFO: Global : 
    [2021/08/13 11:22:15] root INFO:     checkpoints : None
    [2021/08/13 11:22:15] root INFO:     device : cpu
    [2021/08/13 11:22:15] root INFO:     epochs : 1
    [2021/08/13 11:22:15] root INFO:     eval_during_train : True
    [2021/08/13 11:22:15] root INFO:     eval_interval : 1
    [2021/08/13 11:22:15] root INFO:     image_shape : [3, 224, 224]
    [2021/08/13 11:22:15] root INFO:     output_dir : ./output/
    [2021/08/13 11:22:15] root INFO:     pretrained_model : output/ResNet50/best_model
    [2021/08/13 11:22:15] root INFO:     print_batch_step : 1
    [2021/08/13 11:22:15] root INFO:     save_inference_dir : ./inference
    [2021/08/13 11:22:15] root INFO:     save_interval : 1
    [2021/08/13 11:22:15] root INFO:     to_static : False
    [2021/08/13 11:22:15] root INFO:     use_visualdl : True
    [2021/08/13 11:22:15] root INFO: Infer : 
    [2021/08/13 11:22:15] root INFO:     PostProcess : 
    [2021/08/13 11:22:15] root INFO:         class_id_map_file : ./dataset/label_list.txt
    [2021/08/13 11:22:15] root INFO:         name : Topk
    [2021/08/13 11:22:15] root INFO:         topk : 5
    [2021/08/13 11:22:15] root INFO:     batch_size : 10
    [2021/08/13 11:22:15] root INFO:     infer_imgs : dataset/foods/baby_back_ribs/319516.jpg
    [2021/08/13 11:22:15] root INFO:     transforms : 
    [2021/08/13 11:22:15] root INFO:         DecodeImage : 
    [2021/08/13 11:22:15] root INFO:             channel_first : False
    [2021/08/13 11:22:15] root INFO:             to_rgb : True
    [2021/08/13 11:22:15] root INFO:         ResizeImage : 
    [2021/08/13 11:22:15] root INFO:             resize_short : 256
    [2021/08/13 11:22:15] root INFO:         CropImage : 
    [2021/08/13 11:22:15] root INFO:             size : 224
    [2021/08/13 11:22:15] root INFO:         NormalizeImage : 
    [2021/08/13 11:22:15] root INFO:             mean : [0.485, 0.456, 0.406]
    [2021/08/13 11:22:15] root INFO:             order : 
    [2021/08/13 11:22:15] root INFO:             scale : 1.0/255.0
    [2021/08/13 11:22:15] root INFO:             std : [0.229, 0.224, 0.225]
    [2021/08/13 11:22:15] root INFO:         ToCHWImage : None
    [2021/08/13 11:22:15] root INFO: Loss : 
    [2021/08/13 11:22:15] root INFO:     Eval : 
    [2021/08/13 11:22:15] root INFO:         CELoss : 
    [2021/08/13 11:22:15] root INFO:             weight : 1.0
    [2021/08/13 11:22:15] root INFO:     Train : 
    [2021/08/13 11:22:15] root INFO:         CELoss : 
    [2021/08/13 11:22:15] root INFO:             weight : 1.0
    [2021/08/13 11:22:15] root INFO: Metric : 
    [2021/08/13 11:22:15] root INFO:     Eval : 
    [2021/08/13 11:22:15] root INFO:         TopkAcc : 
    [2021/08/13 11:22:15] root INFO:             topk : [1, 5]
    [2021/08/13 11:22:15] root INFO:     Train : 
    [2021/08/13 11:22:15] root INFO:         TopkAcc : 
    [2021/08/13 11:22:15] root INFO:             topk : [1, 5]
    [2021/08/13 11:22:15] root INFO: Optimizer : 
    [2021/08/13 11:22:15] root INFO:     lr : 
    [2021/08/13 11:22:15] root INFO:         decay_epochs : [30, 60, 90]
    [2021/08/13 11:22:15] root INFO:         learning_rate : 0.015
    [2021/08/13 11:22:15] root INFO:         name : Piecewise
    [2021/08/13 11:22:15] root INFO:         values : [0.1, 0.01, 0.001, 0.0001]
    [2021/08/13 11:22:15] root INFO:     momentum : 0.9
    [2021/08/13 11:22:15] root INFO:     name : Momentum
    [2021/08/13 11:22:15] root INFO:     regularizer : 
    [2021/08/13 11:22:15] root INFO:         coeff : 0.0005
    [2021/08/13 11:22:15] root INFO:         name : L2
    [2021/08/13 11:22:16] root INFO: train with paddle 2.1.2 and device CPUPlace
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    [{'class_ids': [3, 5, 4, 2, 1], 'scores': [1.0, 0.0, 0.0, 0.0, 0.0], 'file_name': 'dataset/foods/baby_back_ribs/319516.jpg', 'label_names': ['beef_tartare', 'baklava', 'apple_pie', 'baby_back_ribs', 'beef_carpaccio']}]


运行完成，最后几行会得到结果如下形式：
```
[{'class_ids': [5, 1, 3, 4, 2],
'scores': [0.48433, 0.26765, 0.13903, 0.05609, 0.05162],
'file_name': 'dataset/foods/baby_back_ribs/319516.jpg', 
'label_names': ['baklava', 'beef_carpaccio', 'beef_tartare', 'apple_pie', 'baby_back_ribs']}]
```
可以发现，预测结果不对，准确率很低，但整体的项目流程你已经掌握了！    
训练轮数还有很大提升空间，自行变动参数直到预测正确为止~    

<font size="3px" color="red">恭喜你学会了paddleclas图像分类！</font>

## 最后（！很重要！）    
1. 生成版本     
![](https://ai-studio-static-online.cdn.bcebos.com/adda3ac3b96c403daa68f9f5ec9fd807930d1404f5834317a4f3d7e662243ebc)   
2. 有能力者导出markdown发表到github上      
![](https://ai-studio-static-online.cdn.bcebos.com/a1f8fdad7659463381d8857e9fd63e84648591fc1a264f1a8241029fb825fe78)      
3. 公开项目   
![](https://ai-studio-static-online.cdn.bcebos.com/bc8af42d84914b7a8e882e3b88551aa7a5943cf7351747b2b6df3310be53eb05)    
4. <font color="red">祝大家顺利结业~</font>
