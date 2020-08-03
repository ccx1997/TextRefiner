import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ConvBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """For aster or crnn"""
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer0 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True))

        self.inplanes = 32
        self.layer1 = self._make_layer(32,  3, [2, 2])
        self.layer2 = self._make_layer(64,  4, [2, 2])
        self.layer3 = self._make_layer(128, 6, [2, 1])
        self.layer4 = self._make_layer(256, 6, [2, 1])
        self.layer5 = self._make_layer(512, 3, [2, 1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes, stride=stride),
                    nn.BatchNorm2d(planes))
        layers = []
        layers.append(ConvBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ConvBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def test():
    x = torch.randn(2, 3, 32, 128)
    model = ResNet()
    print(model)
    y = model(x)
    print(y.size())    # Expected to be [2, 512, 1, 32]


if __name__ == '__main__':
    test()
