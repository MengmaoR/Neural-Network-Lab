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

# 修改 kernel_size 为 3
args['kernel_size'] = 3
train.args = args
train.main()
resize_loss, resize_acc = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 修改 kernel_num1 为 12, kernel_num2 为 32
args['kernel_size'] = 5
args['kernel_num1'] = 12
args['kernel_num2'] = 32
train.args = args
train.main()
renum1_loss, renum1_acc = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 修改 kernel_num1 为 24, kernel_num2 为 64
args['kernel_num1'] = 24
args['kernel_num2'] = 64
train.args = args
train.main()
renum2_loss, renum2_acc = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# Plot loss_log
plt.plot(typical_loss, "o", color='blue', alpha=0.2, label='Typical')
plt.plot(resize_loss, "o", color='red', alpha=0.2, label='Kernel Size 3')
plt.plot(renum1_loss, "o", color='orange', alpha=0.2, label='Kernel Num 12, 32')
plt.plot(renum2_loss, "o", color='green', alpha=0.2, label='Kernel Num 24, 64')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.savefig(f'loss_kernels_{args["dataset"]}_{args["model"]}.png', dpi = 600)
plt.clf()

# Plot acc_log
plt.plot(typical_conv, label='Typical')
plt.plot(resize_acc, label='Kernel Size 3')
plt.plot(renum1_acc, label='Kernel Num 12, 32')
plt.plot(renum2_acc, label='Kernel Num 24, 64')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.savefig(f'acc_kernels_{args["dataset"]}_{args["model"]}.png', dpi = 600)