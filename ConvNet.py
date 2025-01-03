import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):   # SE 注意力模块
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y

class ECABlock(nn.Module):  # ECA 注意力模块
    def __init__(self, channels, kernel_size=3):
        super(ECABlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, channels, 1)  # Global Average Pooling
        y = self.conv1d(y.permute(0, 2, 1))  # 1D convolution
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y

class CBAMBlock(nn.Module): # CBAM 注意力模块
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # Channel Attention Module
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention Module
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        batch, channels, height, width = x.size()
        avg_out = self.global_avg_pool(x).view(batch, channels)
        max_out = self.global_max_pool(x).view(batch, channels)
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(batch, channels, 1, 1)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att

class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()

        # parameters
        self.activation = args['activation']
        self.normalization = args['normalization']
        self.dropout = args['dropout']
        self.attention = args['attention']
        self.add_conv = args['add_conv']
        self.kernel_size = args['kernel_size']
        self.kernel_num1 = args['kernel_num1']
        self.kernel_num2 = args['kernel_num2']
        self.dataset = args['dataset']
        
        # layers
        # convolutional 1
        if args['add_conv']:    # add conv
            self.conv0 = nn.Conv2d(3, 6, 3)                     # 6x30x30
            self.conv1 = nn.Conv2d(6, args['kernel_num1'], 3)   # 6x28x28
        else:
            self.conv1 = nn.Conv2d(3, args['kernel_num1'], args['kernel_size']) # 12x28x28

        # normalization 1
        if self.normalization == 'bn':
            self.norm1 = nn.BatchNorm2d(args['kernel_num1'])
        elif self.normalization == 'ln':
            self.norm1 = nn.LayerNorm([args['kernel_num1'], 28, 28])
        elif self.normalization == 'gn':
            self.norm1 = nn.GroupNorm(6, args['kernel_num1'])
        
        # attention 1
        if self.attention == 'se':
            self.att1 = SEBlock(args['kernel_num1'])
        elif self.attention == 'eca':
            self.att1 = ECABlock(args['kernel_num1'])
        elif self.attention == 'cbam':
            self.att1 = CBAMBlock(args['kernel_num1'])

        # pooling 1
        self.pool1 = nn.MaxPool2d(2, 2) # 12x14x14

        # convolutional 2
        self.conv2 = nn.Conv2d(args['kernel_num1'], args['kernel_num2'], args['kernel_size']) # 32x10x10

        # normalization 2
        if self.normalization == 'bn':
            self.norm2 = nn.BatchNorm2d(args['kernel_num2'])
        elif self.normalization == 'ln':
            self.norm2 = nn.LayerNorm([args['kernel_num2'], 10, 10])
        elif self.normalization == 'gn':
            self.norm2 = nn.GroupNorm(16, args['kernel_num2'])
        
        # attention 2
        if self.attention == 'se':
            self.att2 = SEBlock(args['kernel_num2'])
        elif self.attention == 'eca':
            self.att2 = ECABlock(args['kernel_num2'])
        elif self.attention == 'cbam':
            self.att2 = CBAMBlock(args['kernel_num2'])
        
        # pooling 2
        if self.kernel_size == 5:
            self.pool2 = nn.MaxPool2d(2, 2) # 16x5x5
        elif self.kernel_size == 3:
            self.pool2 = nn.MaxPool2d(2, 2, padding=1) # 16x7x7

        # additional convolutional
        if self.kernel_size == 5:
            self.conv3 = nn.Conv2d(args['kernel_num2'], 120, 5)
        elif self.kernel_size == 3:
            self.conv3 = nn.Conv2d(args['kernel_num2'], 120, 7)
        
        # fully connected 1
        self.fc1 = nn.Linear(120,84)

        if self.normalization == 'bn':
            self.norm4 = nn.BatchNorm1d(84)
        elif self.normalization == 'ln':
            self.norm4 = nn.LayerNorm(84)
        elif self.normalization == 'gn':
            self.norm4 = nn.GroupNorm(1, 84)

        # dropout
        if self.dropout:
            self.dropout = nn.Dropout(0.5)

        # fully connected 2
        if self.dataset == 'cifar10':
            self.fc2 = nn.Linear(84, 10)
        elif self.dataset == 'cifar100':
            self.fc2 = nn.Linear(84, 100)

    def activation_layer(self, input):
        if self.activation == 'relu':
            return F.relu(input)
        elif self.activation == 'gelu':
            return F.gelu(input)
        elif self.activation == 'sigmoid':
            return F.sigmoid(input)
        elif self.activation == 'tanh':
            return F.tanh(input)
        else:
            return input
        
    def forward(self, x):
        in_size = x.size(0)
        if self.add_conv:
            out = self.conv0(x)
            out = self.activation_layer(out)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        if self.normalization != 'none' and self.normalization != 'ln':
            out = self.norm1(out)
        out = self.activation_layer(out)
        if self.attention != 'none':
            out = self.att1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        if self.normalization != 'none' and self.normalization != 'ln':
            out = self.norm2(out)
        out = self.activation_layer(out)
        if self.attention != 'none':
            out = self.att2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = out.view(in_size,-1)
        if self.normalization != 'none':
            out = self.norm3(out)
        out = self.fc1(out)
        if self.normalization != 'none':
            out = self.norm4(out)
        out = self.activation_layer(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)

        out = F.log_softmax(out,dim=1)
        return out