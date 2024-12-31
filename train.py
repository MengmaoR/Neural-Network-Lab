import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import ConvNet

args = {
    'model': 'lenet5',          # lenet5
    'dataset': 'cifar100',       # CIFAR10, CIFAR100
    'batch_size': 512,          # 64, 128, 256, 512
    'optimizer': 'adam',        # adam, sgd, adadelta, rmsprop
    'activation': 'relu',       # relu, gelu, sigmoid, tanh
    'normalization': 'none',    # none, bn, ln, gn
    'attention': 'none',        # none, se, eca, cbam
    'dropout': False,           # True, False
    'add_conv': False,          # True, False
    'kernel_size': 5,           # 3, 5
    'kernel_num1': 6,           # 6, 12, 24
    'kernel_num2': 16           # 16, 32, 64
}
total_epoch = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Runing on", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

loss_log = []
acc_log = []

# Data preprocessing
def data_preprocessing():
    if args['dataset'] == 'cifar10':
        if args['model'] == 'lenet5':   
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform),
            batch_size=args['batch_size'], shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transform),
            batch_size=args['batch_size'], shuffle=True
        )
    
    elif args['dataset'] == 'cifar100':
        if args['model'] == 'lenet5':   
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True, transform=transform),
            batch_size=args['batch_size'], shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, transform=transform),
            batch_size=args['batch_size'], shuffle=True
        )

    return train_loader, test_loader

def train(model, train_loader, opt, epoch):
    model.train()    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt.step()
        loss_log.append(loss.item())
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    acc_log.append(acc)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({acc:.2f}%)\n')


def main():
    if args['model'] == 'lenet5':
        model = ConvNet.model(args).to(device)
    
    if args['optimizer'] == 'adam':
        opt = optim.Adam(model.parameters())
    elif args['optimizer'] == 'sgd':
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif args['optimizer'] == 'adadelta':
        opt = optim.Adadelta(model.parameters())
    elif args['optimizer'] == 'rmsprop':
        opt = optim.RMSprop(model.parameters())
    
    train_loader, test_loader = data_preprocessing()

    for epoch in range(total_epoch):
        train(model, train_loader, opt, epoch+1)
        test(model, test_loader)
    
if __name__ == '__main__':
    main()