import train
import matplotlib.pyplot as plt

args = train.args
train.total_epoch = 10
args['dataset'] = 'mnist'

# 经典 LeNet-5，无归一化，batch_size = 64
train.args = args
train.main()
none_64_loss, none_64_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# bn 归一化，batch_size = 64
args['normalization'] = 'bn'
train.args = args
train.main()
bn_64_loss, bn_64_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# ln 归一化，batch_size = 64
args['normalization'] = 'ln'
train.args = args
train.main()
ln_64_loss, ln_64_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# gn 归一化，batch_size = 64
args['normalization'] = 'gn'
train.args = args
train.main()
gn_64_loss, gn_64_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# 无归一化，batch_size = 128
args['batch_size'] = 128
args['normalization'] = 'none'
train.args = args
train.main()
none_128_loss, none_128_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# bn 归一化，batch_size = 128
args['normalization'] = 'bn'
train.args = args
train.main()
bn_128_loss, bn_128_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# ln 归一化，batch_size = 128
args['normalization'] = 'ln'
train.args = args
train.main()
ln_128_loss, ln_128_conv = train.loss_log, train.acc_log
train.loss_log, train.acc_log = [], []

# gn 归一化，batch_size = 128
args['normalization'] = 'gn'
train.args = args
train.main()
gn_128_loss, gn_128_conv = train.loss_log, train.acc_log

# Plot loss_log
plt.plot(none_64_loss, "o", color='blue', alpha=0.5, label='None 64')
plt.plot(bn_64_loss, "o", color='red', alpha=0.5, label='BN 64')
plt.plot(ln_64_loss, "o", color='orange', alpha=0.5, label='LN 64')
plt.plot(gn_64_loss, "o", color='green', alpha=0.5, label='GN 64')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.savefig(f'loss_norm_{args["dataset"]}_{args["model"]}_64.png', dpi = 600)
plt.clf()

plt.plot(none_128_loss, "o", color='blue', alpha=0.5, label='None 128')
plt.plot(bn_128_loss, "o", color='red', alpha=0.5, label='BN 128')
plt.plot(ln_128_loss, "o", color='orange', alpha=0.5, label='LN 128')
plt.plot(gn_128_loss, "o", color='green', alpha=0.5, label='GN 128')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.savefig(f'loss_norm_{args["dataset"]}_{args["model"]}_128.png', dpi = 600)
plt.clf()

# Plot acc_log
plt.plot(none_64_conv, label='None 64')
plt.plot(bn_64_conv, label='BN 64')
plt.plot(ln_64_conv, label='LN 64')
plt.plot(gn_64_conv, label='GN 64')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.savefig(f'acc_norm_{args["dataset"]}_{args["model"]}_64.png', dpi = 600)
plt.clf()

plt.plot(none_128_conv, label='None 128')
plt.plot(bn_128_conv, label='BN 128')
plt.plot(ln_128_conv, label='LN 128')
plt.plot(gn_128_conv, label='GN 128')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.savefig(f'acc_norm_{args["dataset"]}_{args["model"]}_128.png', dpi = 600)