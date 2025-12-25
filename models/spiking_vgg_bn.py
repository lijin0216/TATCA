import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer
from modules.layers import TATCA, TA, CA 

__all__ = [
    'SpikingVGGBN', 'spiking_vgg11_bn', 'spiking_vgg13_bn', 'spiking_vgg16_bn', 'spiking_vgg19_bn'
]

cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class SpikingVGGBN(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, T: int = 6, **kwargs):
        super(SpikingVGGBN, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)

        self.part1 = nn.Sequential(
            self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs),
            self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs),
            self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
        )
        

        tatca_input_channels = 256
        self.tatca_module = TATCA(kernel_size_t=3, T=T, channel=tatca_input_channels)  
        #self.tatca_module = TA(kernel_size_t=3, kernel_size_c=3, T=T, channel=tatca_input_channels)  
        #self.tatca_module = CA(kernel_size_t=3, kernel_size_c=3, T=T, channel=tatca_input_channels)  

        self.part2 = nn.Sequential(
            self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs),
            self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(512, num_classes) 
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout, neuron, **kwargs):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                layers.append(nn.BatchNorm2d(x))
                layers.append(neuron(**kwargs))
                layers.append(layer.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        raise NotImplementedError("This model must be trained with the custom training loop.")


def spiking_vgg11_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, T: int = 6, **kwargs):
    return SpikingVGGBN('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, T=T, **kwargs)


def spiking_vgg13_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, T: int = 6, **kwargs):
    return SpikingVGGBN('VGG13', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, T=T, **kwargs)


def spiking_vgg16_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, T: int = 6, **kwargs):
    return SpikingVGGBN('VGG16', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, T=T, **kwargs)


def spiking_vgg19_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, T: int = 6, **kwargs):
    return SpikingVGGBN('VGG19', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, T=T, **kwargs)