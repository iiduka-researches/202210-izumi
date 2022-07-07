from typing import Sequence, Union
import torch
import torch.nn as nn
Channels = Union[int, Sequence[int]]

__all__ = ['ResNetCIFAR', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


def conv3x3(in_channels: Channels, out_channels: Channels, stride: int = 1, groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
    """
    Make a 3x3 convolution with padding

    Parameters
    ----------
    in_channels: int
        最初の畳み込み層への入力のchannel数（色数のようなもの）。
    out_channels: int
        最初の畳み込み層の出力のchannel数。
    stride: int, optional, default 1
        controls the stride for the cross-correlation.
        相互相関関数(カーネル)を適用する際に窓を動かす幅。
    groups: int, optional, default 1
        controls the connections between inputs and outputs.
        in_channels and out_channels must both be divisible by groups.
        結合を決めるパラメータ群(層と層の結合)の数。
        in_channelsとout_channelsを割り切れる(公約数である)必要がある。
    dilation: int, optional, default 1
        controls the spacing between the kernel points; also known as the à trous algorithm.
        和を計算する窓の中心からの距離。dilation = 1は通常の畳み込み。

    Returns
    -------
    conv: torch.nn.Conv2d

    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, groups=groups, bias=False)
    return conv


def conv1x1(in_channels: int or tuple, out_channels: int or tuple, stride=1):
    """
    Make 1x1 convolution

    Parameters
    ---------
    in_channels: int
        最初の畳み込み層への入力のchannel数（色数のようなもの）。
    out_channels: int
        最初の畳み込み層の出力のchannel数。
    stride: int, optional, default 1
        controls the stride for the cross-correlation.
        相互相関関数(カーネル)を適用する際に窓を動かす幅。

    Returns
    -------
    conv: torch.nn.Conv2d

    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    return conv


class BasicBlock(nn.Module):
    """
    Attributes
    ----------
    conv1 = torch.nn.Conv2d
    bn1 = torch.nn.Module
    relu = torch.nn.ReLU
    conv2 = torch.nn.Conv2d
    bn2: torch.nn.Module
    stride: int, optional, default 1
        相互相関関数(カーネル)を適用する際に窓を動かす幅。
    """
    expansion = 1  # What?
    __constants__ = ['downsample']  # What?

    def __init__(self, in_channels: Channels, out_channels: Channels, stride=1, downsample=None, groups=1,
                 base_width=32, dilation=1, norm_layer=None):
        """
        Parameters
        ----------
        in_channels: int or tuple of int
            最初の畳み込み層への入力のchannel数（色数のようなもの）。
        out_channels: int or tuple of int
            最初の畳み込み層の出力のchannel数。
        stride: int, optional, default 1
            相互相関関数(カーネル)を適用する際に窓を動かす幅。
        downsample: torch.nn.Module, optional, default None
        groups: int, optional, default 1,
            結合を決めるパラメータ群(層と層の結合)の数。
            in_channelsとout_channelsを割り切れる(公約数である)必要がある。
        base_width: int, optional, default 32,
        dilation: int, optional, default 1
            和を計算する窓の中心からの距離。dilation = 1は通常の畳み込み。
        norm_layer: torch.nn.Module, optional, default None
            バッチ正則化のための層。

        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 32:
            raise ValueError('BasicBlock only supports groups=1 and base_width=32')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            ブロックへの入力

        Returns
        -------
        out: torch.Tensor
            ブロックの出力

        """
        identity = x

        """
        1層目
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        """
        2層目
        """
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        """
        shortcut connection
        """
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=32, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 32.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    """
    Attributes
    ----------
    _norm_layer: torch.nn.Module
        バッチ正則化のための層。
    in_channels: int or tuple of int

    dilation: int
        和を計算する窓の中心からの距離。dilation = 1は通常の畳み込み。
    groups:
        結合を決めるパラメータ群(層と層の結合)の数。
        in_channelsとout_channelsを割り切れる(公約数である)必要がある。
    width_per_group: int
    conv1: torch.nn.Conv2d
    bn1: torch.nn.Module
    relu: torch.nn.ReLU
    maxpool: torch.nn.MaxPool2d
        ResNet
    layer1: torch.nn.Sequential
        ResNetを構成する層群1つ目。
    layer2: torch.nn.Sequential
        ResNetを構成する層群2つ目。
    layer3: torch.nn.Sequential
        ResNetを構成する層群3つ目。
    avgpool: torch.nn.AdaptiveAvgPool2d
        ResNetの最後のPooling層。
    fc: torch.nn.Linear


    """
    def __init__(self, block: nn.Module, layers: list, num_classes: int = 10, zero_init_residual: bool = False,
                 groups=1, width_per_group=32, replace_stride_with_dilation=None, norm_layer=None) -> None:
        """
        Parameters
        ----------
        block: torch.nn.Module
            ResNetのブロック。
        layers: list of int

        num_classes: int, optional, default 10
            分類class数
        zero_init_residual: bool, optional, default False

        groups: int, optional, default 1
            結合を決めるパラメータ群(層と層の結合)の数。
            in_channelsとout_channelsを割り切れる(公約数である)必要がある。
        width_per_group: int, optional, default 32

        replace_stride_with_dilation: list of bool, optional, default None

        norm_layer: torch.nn, optional, default None
            バッチ正則化のための層。
        """
        super(ResNetCIFAR, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: nn.Module, out_channels: int or tuple, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """
        Make ResNet

        Parameters
        ----------
        block: torch.nn.Module
            ResNetのblock。
        out_channels:
            最初の畳み込み層の出力のchannel数
        blocks: int
            blockの個数。
        stride: int, optional, default 1
            相互相関関数(kernel)を適用する際に窓を動かす幅。
        dilate: bool, optional, default False
            dilationを適用するかどうか。

        Returns
        -------
        sequential: torch.nn.Sequential
            layersから構成されるResNet
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_channels, out_channels * block.expansion, stride),
                                       norm_layer(out_channels * block.expansion))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        sequential = nn.Sequential(*layers)

        return sequential

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate forward

        Parameters
        ----------
        x: torch.Tensor
            ResNetへの入力。

        Returns
        -------
        x: torch.Tensor
            ResNetの出力。

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


def _resnet(block: nn.Module, layers: list, **kwargs) -> ResNetCIFAR:
    """
    Build ResNet

    Parameters
    ----------
    block: torch.nn.Module
        ResNetを構成するblock。
    layers: list of int
        ResNetを構成する格層群の層数。
    Returns
    -------
    model: ResNetCIFAR
        ResNet model
    """
    model = ResNetCIFAR(block, layers, **kwargs)
    return model


def resnet20(**kwargs) -> ResNetCIFAR:
    """
    Build 20-layer ResNet (ResNet-20) for CIFAR-10 dataset.

    Parameters
    ----------
    **kwargs

    Returns
    -------
    model: ResNetCIFAR
        ResNet-20 model
    """
    model = _resnet(BasicBlock, [3, 3, 3], **kwargs)

    return model


def resnet32(**kwargs) -> ResNetCIFAR:
    """
    Build 32-layer ResNet (ResNet-32) for CIFAR-10 dataset.

    Parameters
    ----------
    **kwargs

    Returns
    -------
    model: ResNetCIFAR
        ResNet-32 model
    """
    model = _resnet(BasicBlock, [5, 5, 5], **kwargs)

    return model


def resnet44(**kwargs) -> ResNetCIFAR:
    """
    Build 44-layer ResNet (ResNet-44) for CIFAR-10 dataset.

    Parameters
    ----------
    **kwargs

    Returns
    -------
    model: ResNetCIFAR
        ResNet-44 model
    """
    model = _resnet(BasicBlock, [7, 7, 7], **kwargs)

    return model


def resnet56(**kwargs) -> ResNetCIFAR:
    """
    Build 56-layer ResNet (ResNet-56) for CIFAR-10 dataset.

    Parameters
    ----------
    **kwargs

    Returns
    -------
    model: ResNetCIFAR
        ResNet-56 model
    """
    model = _resnet(BasicBlock, [9, 9, 9], **kwargs)

    return model


def resnet110(**kwargs) -> ResNetCIFAR:
    """
    Build 110-layer ResNet (ResNet-110) for CIFAR-10 dataset.

    Parameters
    ----------
    **kwargs

    Returns
    -------
    model: ResNetCIFAR
        ResNet-110 model
    """
    model = _resnet(BasicBlock, [18, 18, 18], **kwargs)

    return model
