import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- InceptionNet or GoogLenet ----------------------------------------
# the class "Inception"  and "MyInceptionNet" are adapted by torchvision.models.googlenet
# and the two related paper

# 我们发现, 在每个Inception后的 concat 后多加一层"Relu" , 在epoch < 24的训练中,效果有明显提升, 详见报告

class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3red: int,
            ch3x3: int,
            ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
            if_efficiency=False,  # 参考 rethink Inception, 思索卷积核的极限
            if_SE_block=False
    ) -> None:
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),  # 降卷积核维度 / 节省计算
            nn.BatchNorm2d(ch1x1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3)
        )

        if if_efficiency:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
                # 更高效的方式: 用两波 3*3 完美代替 5*5
                nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
                nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch5x5)
            )
        else:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
                # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
                # 但,我们只是想试验，也不用pytorch的预训练，直白还原
                nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
                nn.BatchNorm2d(ch5x5)
            )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj)
        )
        self.if_SE_block = if_SE_block
        if if_SE_block:
            self.se_block = SEblock(ch1x1 + ch3x3 + ch5x5 + pool_proj)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        # 我们发现，在每次Inception网络后 加与不加 Relu 激活函数,在我们的试验中有明显区别
        # 加完relu,训练效果更好，算是我们对 原torchvision.models.googlenet 中Inception的改进
        if self.if_SE_block:
            return F.relu( self.se_block(torch.cat(outputs, 1)) )
        else:
            return F.relu(torch.cat(outputs, 1))
            # return torch.cat(outputs, 1)


class MyInceptionNet(nn.Module):
    # 借鉴GoogLeNet而来 , 参考; 为了和Alexnet保持同步,本次也选择5层深度的卷积层
    def __init__(
            self,
            num_classes: int = 10,
            if_efficiency=False,  # Add SE-block or not
            if_SE=False
    ):
        super(MyInceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # inception 参数 [ in_channels , ch1x1 ,ch3x3red: reduction ,ch3x3: int,ch5x5red: reduction ,ch5x5 ,pool_proj ]
        self.inception1 = Inception(16, 16, 24, 32, 4, 8, 8, if_efficiency=if_efficiency, if_SE_block=if_SE)
        self.inception2 = Inception(64, 32, 32, 48, 8, 24, 16, if_efficiency,
                                    if_SE)  # output channels: [ 120 * 32 *32 ]
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H W砍一半 [120 * 16 * 16]

        self.inception3 = Inception(120, 48, 24, 52, 4, 12, 16, if_efficiency, if_SE)
        self.inception4 = Inception(128, 40, 28, 56, 6, 16, 16, if_efficiency, if_SE)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # H W砍一半 [128 * 8 * 8]

        self.inception5 = Inception(128, 64, 40, 80, 8, 32, 32, if_efficiency, if_SE)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 压成 C*1*1
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(208, num_classes)

    def forward(self, x):
        # N x 3 x 32 x 32
        x = F.relu(self.bn1(self.conv1(x)))  # 经典三件套
        # N x 16 x 32 x 32
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool1(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.maxpool2(x)
        x = self.inception5(x)

        x = self.avgpool(x)
        # N x 208 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 208
        x = self.dropout(x)
        x = self.fc(x)
        # N x 10 (num_classes)
        return x


# ------------------ SE block SE-ResNet / SE-InceptionNet ----------------------------------
# the following class are adapted by Hu Jie's paper <Squeeze-and-Excitation Networks>
class SEblock(nn.Module):
    def __init__(self, input_channel: int, r: int = 8):
        super(SEblock, self).__init__()
        self.avgPool_glb = nn.AdaptiveAvgPool2d((1, 1))  # C * H * W -> C
        self.fc1 = nn.Linear(input_channel, input_channel // r)  # 降维
        self.fc2 = nn.Linear(input_channel // r, input_channel)  # 生维
        return

    def forward(self, x):
        # x : [BatchSize * C * H * W]
        # squeeze
        z = self.avgPool_glb(x)  # [BS * C ]
        # excitation
        z = z.squeeze(-1).squeeze(-1)
        z = F.relu(self.fc1(z))  # [BS * C/r]
        z = torch.sigmoid(self.fc2(z))  # [BS * C ]
        # rescaling
        output = torch.einsum('ajbc,aj->ajbc ', x, z)  # [B , C , H, W]
        return output


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SE_BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,  # input channels = planes * extension
            planes: int,  # output channels
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None,
    ) -> None:
        super(SE_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.se_block = SEblock(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se_block(out)  # 就加一层

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SE_Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,  # input channels
            planes: int,  # output channels
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None
    ) -> None:
        super(SE_Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  # 压到 output channel的维度
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)  # 降维
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # 升高回去维度
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.seBlock = SEblock(input_channel=planes * self.expansion)

    def forward(self, x):
        identity = x  #

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.seBlock(out)

        if self.downsample is not None:  # stride == 2 ,则x需要降采样，H*W -> 0.5*H 0.5*W   C不变
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SE_ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            num_classes=10,
            groups: int = 1,
            width_per_group: int = 64,
            zero_init_residual=False
    ) -> None:
        super(SE_ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 16  # 第一层 output_channels
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  # 改
        self.bn1 = self._norm_layer(self.inplanes)
        self.se_block = SEblock(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # half

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # 此处 stride = 2 降采样（少重复卷）
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

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
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.se_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)