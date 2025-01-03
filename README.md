# Neural-Network-Lab

## 环境配置
```shell
pip install -r requirements.txt
```

## 代码执行
对于每次训练，训练的 epoch 数和训练数据集需在每个训练代码文件的文件头处手动修改，其余部分均已自动配置。
需修改的部分如下：
```python
args = train.args
train.total_epoch = 30          # 训练的 epoch 数
args['dataset'] = 'cifar100'    # 训练数据集
```

### 作业一
```shell
python ./PA1/test_layer.py      # 网络架构分析
python ./PA1/test_kernel.py     # 卷积核参数分析
python ./PA1/test_activate.py   # 激活函数分析
python ./PA1/test_normalize.py  # 归一化方法分析
```

### 作业二
```shell
python ./PA2/test_optimizer.py  # 优化器分析
```

### 作业三
```shell
python ./PA3/test_attention.py  # 注意力机制分析
```

## 训练结果
每个 PA 的训练结果会保存在其对应文件夹下的 img 文件夹中，包含训练过程中的 loss 和 accuracy 图像，并按其分析类型，使用模型和训练数据集进行命名，例如：
```shell
# 作业一中网络结构分析的训练结果
PA1/img/acc_layer_cifar100_lenet5.png
PA1/img/loss_layer_cifar100_lenet5.png
```