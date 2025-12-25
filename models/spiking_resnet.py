import torch.nn as nn
import torch
from spikingjelly.clock_driven import layer
from modules.layers import TATCA, TA, CA 

__all__ = [
    'PreActResNet', 'spiking_resnet18', 'spiking_resnet34'
]

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, dropout, neuron: callable = None, **kwargs):
        super(PreActBlock, self).__init__()
        whether_bias = True
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=whether_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = layer.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1, bias=whether_bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, bias=whether_bias)
        else:
            self.shortcut = nn.Sequential()

        self.relu1 = neuron(**kwargs)
        self.relu2 = neuron(**kwargs)

    def forward(self, x):
        x = self.relu1(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        out = out + self.shortcut(x)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, dropout, neuron: callable = None, **kwargs):
        super(PreActBottleneck, self).__init__()
        whether_bias = True

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=whether_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=whether_bias)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = layer.Dropout(dropout)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, padding=0, bias=whether_bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, bias=whether_bias)
        else:
            self.shortcut = nn.Sequential()

        self.relu1 = neuron(**kwargs)
        self.relu2 = neuron(**kwargs)
        self.relu3 = neuron(**kwargs)

    def forward(self, x):
        x = self.relu1(self.bn1(x))

        out = self.conv1(x)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.dropout(self.relu3(self.bn3(out))))

        out = out + self.shortcut(x)

        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes, dropout, neuron: callable = None, T: int = 6, **kwargs):
        super(PreActResNet, self).__init__()
        self.num_blocks = num_blocks
        self.data_channels = kwargs.get('c_in', 3)
        self.init_channels = 64

        self.part1 = nn.Sequential(
            nn.Conv2d(self.data_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            self._make_layer(block, 64, num_blocks[0], 1, dropout, neuron, **kwargs),
            self._make_layer(block, 128, num_blocks[1], 2, dropout, neuron, **kwargs)
        )
        
        tatca_input_channels = 128 * block.expansion
        self.tatca_module = TATCA(kernel_size_t=3, T=T, channel=tatca_input_channels)
        #self.tatca_module = TA(kernel_size_t=3, kernel_size_c=3, T=T, channel=tatca_input_channels)  
        #self.tatca_module = CA(kernel_size_t=3, kernel_size_c=3, T=T, channel=tatca_input_channels)  

        self.part2 = nn.Sequential(
            self._make_layer(block, 256, num_blocks[2], 2, dropout, neuron, **kwargs),
            self._make_layer(block, 512, num_blocks[3], 2, dropout, neuron, **kwargs),
            nn.BatchNorm2d(512 * block.expansion),
            neuron(**kwargs),  
            nn.AvgPool2d(4),
            nn.Flatten(),
            layer.Dropout(dropout),
            nn.Linear(512 * block.expansion, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout, neuron, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.init_channels, out_channels, s, dropout, neuron, **kwargs))
            self.init_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        raise NotImplementedError("This model must be trained with the custom hybrid loop .")



def spiking_resnet18(neuron: callable = None, num_classes=10,  neuron_dropout=0, T: int = 6, **kwargs):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes, neuron_dropout, neuron=neuron, T=T, **kwargs)


def spiking_resnet34(neuron: callable = None, num_classes=10,  neuron_dropout=0, T: int = 6, **kwargs):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes, neuron_dropout, neuron=neuron, T=T, **kwargs)
