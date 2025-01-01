import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train
import matplotlib.pyplot as plt

args = train.args
train.total_epoch = 30
args['dataset'] = 'cifar100'

# 经典 LeNet-5，使用 relu
train.args = args
train.main()
relu_loss, relu_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 使用 gelu
args['activation'] = 'gelu'
train.args = args
train.main()
gelu_loss, gelu_acc = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 使用 sigmoid
args['activation'] = 'sigmoid'
train.args = args
train.main()
sigmoid_loss, sigmoid_acc = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 使用 tanh
args['activation'] = 'tanh'
train.args = args
train.main()
tanh_loss, tanh_acc = train.loss_log, train.acc_log

# Plot loss_log
plt.plot(relu_loss, "o", color='blue', alpha=0.2, label='ReLU')
plt.plot(gelu_loss, "o", color='red', alpha=0.2, label='GELU')
plt.plot(sigmoid_loss, "o", color='orange', alpha=0.2, label='Sigmod')
plt.plot(tanh_loss, "o", color='green', alpha=0.2, label='Tanh')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.savefig(f'./PA1/img/loss_activate_{args["dataset"]}_{args["model"]}.png', dpi = 600)
plt.clf()

# Plot acc_log
plt.plot(relu_conv, label='ReLU')
plt.plot(gelu_acc, label='GELU')
plt.plot(sigmoid_acc, label='Sigmod')
plt.plot(tanh_acc, label='Tanh')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.savefig(f'./PA1/img/acc_activate_{args["dataset"]}_{args["model"]}.png', dpi = 600)