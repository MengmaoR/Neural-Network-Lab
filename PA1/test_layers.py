import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train
import matplotlib.pyplot as plt

args = train.args
train.total_epoch = 30
args['dataset'] = 'cifar10'

# 经典 LeNet-5
train.args = args
train.main()
typical_loss, typical_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 增加一个卷积层
args['add_conv'] = True
train.args = args
train.main()
add_conv_loss, add_conv_acc = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 再增加 dropout
args['dropout'] = True
train.args = args
train.main()
both_loss, both_acc = train.loss_log, train.acc_log

# 只添加 dropout，无额外卷积层
train.loss_log, train.acc_log = [], []
args['add_conv'] = False
train.args = args
train.main()
dropout_loss, dropout_acc = train.loss_log, train.acc_log

# Plot loss_log
plt.plot(typical_loss, "o", color='blue', alpha=0.2, label='Typical')
plt.plot(add_conv_loss, "o", color='red', alpha=0.2, label='Add Conv')
plt.plot(dropout_loss, "o", color='orange', alpha=0.2, label='Dropout')
plt.plot(both_loss, "o", color='green', alpha=0.2, label='Both')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.savefig(f'./PA1/img/loss_layers_{args["dataset"]}_{args["model"]}.png', dpi = 600)
plt.clf()

# Plot acc_log
plt.plot(typical_conv, label='Typical')
plt.plot(add_conv_acc, label='Add Conv')
plt.plot(dropout_acc, label='Dropout')
plt.plot(both_acc, label='Both')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.savefig(f'./PA1/img/acc_layers_{args["dataset"]}_{args["model"]}.png', dpi = 600)