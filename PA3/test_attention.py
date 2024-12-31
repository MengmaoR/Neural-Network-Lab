import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train
import matplotlib.pyplot as plt

args = train.args
train.total_epoch = 30
args['dataset'] = 'cifar10'

# 经典 Lenet-5，无注意力机制
train.args = args
train.main()
typical_loss, typical_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 引入 se 注意力模块
args['attention'] = 'se'
train.args = args
train.main()
se_loss, se_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 引入 eca 注意力模块
args['attention'] = 'eca'
train.args = args
train.main()
eca_loss, eca_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 引入 cbam 注意力模块
args['attention'] = 'cbam'
train.args = args
train.main()
cbam_loss, cbam_conv = train.loss_log, train.acc_log

# Plot loss_log
plt.plot(typical_loss, "o", color='blue', alpha=0.5, label='Typical')
plt.plot(se_loss, "o", color='red', alpha=0.5, label='SE')
plt.plot(eca_loss, "o", color='orange', alpha=0.5, label='ECA')
plt.plot(cbam_loss, "o", color='green', alpha=0.5, label='CBAM')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.savefig(f'./PA3/img/loss_att_{args["dataset"]}_{args["model"]}.png', dpi = 600)
plt.clf()

# Plot acc_log
plt.plot(typical_conv, label='Typical')
plt.plot(se_conv, label='SE')
plt.plot(eca_conv, label='ECA')
plt.plot(cbam_conv, label='CBAM')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.savefig(f'./PA3/img/acc_att_{args["dataset"]}_{args["model"]}.png', dpi = 600)
plt.clf()