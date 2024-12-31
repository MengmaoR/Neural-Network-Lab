import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train
import matplotlib.pyplot as plt

args = train.args
train.total_epoch = 30
args['dataset'] = 'cifar10'

# Adam 优化器
train.args = args
train.main()
adam_loss, adam_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# SGD 优化器
args['optimizer'] = 'sgd'
train.args = args
train.main()
sgd_loss, sgd_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# Adadelta 优化器
args['optimizer'] = 'adadelta'
train.args = args
train.main()
adadelta_loss, adadelta_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# RMSprop 优化器
args['optimizer'] = 'rmsprop'
train.args = args
train.main()
rmsprop_loss, rmsprop_conv = train.loss_log, train.acc_log

# Plot loss_log
plt.plot(adam_loss, "o", color='blue', alpha=0.5, label='Adam')
plt.plot(sgd_loss, "o", color='red', alpha=0.5, label='SGD')
plt.plot(adadelta_loss, "o", color='orange', alpha=0.5, label='Adadelta')
plt.plot(rmsprop_loss, "o", color='green', alpha=0.5, label='RMSprop')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.savefig(f'./PA2/loss_opt_{args["dataset"]}_{args["model"]}.png', dpi = 600)
plt.clf()

# Plot acc_log
plt.plot(adam_conv, label='Adam')
plt.plot(sgd_conv, label='SGD')
plt.plot(adadelta_conv, label='Adadelta')
plt.plot(rmsprop_conv, label='RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.savefig(f'./PA2/acc_opt_{args["dataset"]}_{args["model"]}.png', dpi = 600)
plt.clf()