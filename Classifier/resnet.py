import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # one kernel make one feature map
        # kernel 厚度 = channel 數
        self.is_changed = in_channels != out_channels
        self.trans = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        
    def forward(self, x):
        f_x = self.conv1(x)
        f_x = self.bn1(f_x)
        f_x = self.relu(f_x)
        f_x = self.conv2(f_x)
        f_x = self.bn2(f_x)

        if self.is_changed:
            x = self.trans(x)

        x = f_x + x
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=4):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=dilation*out_channels, kernel_size=1, padding=0)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * dilation)

        self.relu = nn.ReLU(inplace=True)
        
        # one kernel make one feature map
        # kernel 厚度 = channel 數
        self.is_changed = in_channels != (out_channels*dilation)
        self.trans = nn.Conv2d(in_channels, out_channels * dilation, kernel_size=1, stride=stride)
    
    
    def forward(self, x):
       
        f_x = self.conv1(x)
        f_x = self.bn1(f_x)
        f_x = self.relu(f_x)
        f_x = self.conv2(f_x)
        f_x = self.bn2(f_x)
        f_x = self.relu(f_x)
        f_x = self.conv3(f_x)
        f_x = self.bn3(f_x)

        if self.is_changed:
            x = self.trans(x)
 
        x = f_x + x
        x = self.relu(x)
        return x


class _ResNet(nn.Module):

    def __init__(self, block, block_cnts, dilation=1):
        super(_ResNet, self).__init__()

        self.in_channels = 64
        self.out_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._layer(block, block_cnts[0], dilation, self.in_channels, self.out_channels, stride=1)
        self.layer2 = self._layer(block, block_cnts[1], dilation, self.in_channels, self.out_channels, stride=2)
        self.layer3 = self._layer(block, block_cnts[2], dilation, self.in_channels, self.out_channels, stride=2)
        self.layer4 = self._layer(block, block_cnts[3], dilation, self.in_channels, self.out_channels, stride=2)

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.dense = nn.Linear(in_features=self.in_channels, out_features=10)
        self.softmax = nn.Softmax(dim=1)
        self.flatern = nn.Flatten(start_dim=1)


    def _layer(self, block, block_cnt, dilation, in_channels, out_channels, stride):
        # in_channels: param of previous block output channel
        # out_channels: param of current block input channel
       
        blocks = []
        blocks.append(
            block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        )
        
        for cnt in range(1, block_cnt):
            b = block(in_channels=dilation * out_channels, out_channels=out_channels)
            blocks.append(b)

        self.in_channels = out_channels * dilation 
        self.out_channels = out_channels * 2
       
        return nn.Sequential(*blocks)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg(x)
  
        x = self.flatern(x)
        x = self.dense(x)

        x = self.softmax(x)
        return x


def load_checkpoint(filepath, block, block_cnt, dilation):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = _ResNet(block, block_cnt, dilation)
    model.load_state_dict(checkpoint['model_stat'])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.load_state_dict(checkpoint['optimizer_stat'])

    return model, optimizer