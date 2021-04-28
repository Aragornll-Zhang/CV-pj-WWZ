import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms


class SEblock(nn.Module):
    def __init__(self,input_channel: int, r: int = 8):
        super(SEblock, self).__init__()
        self.avgPool_glb = nn.AdaptiveAvgPool2d((1,1)) # C * H * W -> C
        self.fc1 = nn.Linear(input_channel , input_channel//r) # 降维
        self.fc2 = nn.Linear( input_channel//r ,input_channel) # 生维
        return

    def forward(self,x):
        # x : [BatchSize * C * H * W]
        # squeeze
        z = self.avgPool_glb(x) # [BS * C ]
        # excitation
        z = z.squeeze(-1).squeeze(-1)
        z = F.relu(self.fc1(z))  # [BS * C/r]
        z = torch.sigmoid(self.fc2(z)) # [BS * C ]
        # rescaling
        output = torch.einsum('ajbc,aj->ajbc ', x, z) # [B , C , H, W]
        return output

# model
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#

class SE_BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int, # input channels = planes * extension
        planes: int, # output channels
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
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
        out = self.se_block(out) # 就加一层

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
        planes: int, # output channels
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None
    ) -> None:
        super(SE_Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups # 压到 output channel的维度
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width) # 降维
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion) # 升高回去维度
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.seBlock = SEblock(input_channel=planes * self.expansion )

    def forward(self, x):
        identity = x #

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.seBlock(out)

        if self.downsample is not None: # stride == 2 ,则x需要降采样，H*W -> 0.5*H 0.5*W   C不变
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int, # input channels = planes * extension
        planes: int, # output channels
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super(BasicBlock, self).__init__()
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


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)



        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,  # input channels
        planes: int, # output channels
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None
    ) -> None:
        super(Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups # 压到 output channel的维度
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width) # 降维
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion) # 升高回去维度
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x #

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: # stride == 2 ,则x需要降采样，H*W -> 0.5*H 0.5*W   C不变
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        groups: int = 1,
        width_per_group: int = 64,
        zero_init_residual = False
    ) -> None:
        super(ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 16 # 第一层 output_channels
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) # 改
        self.bn1 = self._norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # half

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilate=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,dilate=1)

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
        # if dilate:
        #     self.dilation *= stride
        #     stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion #  input_channels  =  planes * block.expansion (自动升维度)
        # block 输出维度为 planes * self.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer)) # planes

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class SE_ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        groups: int = 1,
        width_per_group: int = 64,
        zero_init_residual = False
    ) -> None:
        super(SE_ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 16 # 第一层 output_channels
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) # 改
        self.bn1 = self._norm_layer(self.inplanes)
        self.se_block = SEblock(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # half

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2) # 此处 stride = 2 降采样（少重复卷）
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





def train(model, train_X, loss_func, optimizer , if_expand = False):
    model.train()
    total_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (x, y) in enumerate(train_X):
        x = x.to(device)
        y = y.to(device)
        if if_expand:
            for new_x in expand_train_dat(X=x , expand_factor=10):
                # forward
                outputs = model(new_x)
                loss = loss_func(outputs,y)
                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if (i + 1) % 10 == 0:
                    print("Step [{}/{}] Train Loss: {:.4f}".format(i + 1, len(train_X), loss.item()))
        else:
            outputs = model(x)
            loss = loss_func(outputs, y)
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print("Step [{}/{}] Train Loss: {:.4f}".format(i + 1, len(train_X), loss.item()))

    print(total_loss / len(train_X))
    return total_loss / len(train_X)


def predict(model,test_X):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i , (x,y) in enumerate(test_X):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            if i == 0:
                pred_y = torch.argmax(output, axis=1)
                true_y = y
            else:
                pred_y = torch.cat( (pred_y,torch.argmax(output, axis=1) ) , dim=-1)
                true_y = torch.cat( (true_y , y), dim=-1)

    C = confusion_matrix( y_pred = np.array(pred_y.to('cpu')) , y_true= np.array(true_y.to('cpu'))  )
    return C





if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(    mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])



    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, 4),
    #     transforms.ToTensor(),
    #     transforms.Normalize( mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    # ])

    train_dat = torchvision.datasets.CIFAR10(root='F:\python_AI\cv\cifar-10-python', train=True,download=False ,transform=transform)
    trainLoader = data.DataLoader(train_dat, batch_size=32,shuffle=True) # num_workers=2

    test_dat = torchvision.datasets.CIFAR10(root='F:\python_AI\cv\cifar-10-python', train=False,download=False ,transform=transform)
    testLoader = data.DataLoader(test_dat, batch_size=32,drop_last=False) # num_workers=2

    # for i in testLoader:
    #     print(len(i))
    #     print(i[-1])
    #     print('okk')
    #     break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Resnet_model = ResNet(block=BasicBlock , layers=[1,1,1], num_classes=10) # ResNet 6*1 + 2
    # Resnet_model = SE_ResNet(block=SE_BasicBlock , layers=[1,1,1], num_classes=10) # SE-ResNet
    Resnet_model = SE_ResNet(block=SE_Bottleneck , layers=[1,1,1], num_classes=10) # SE-ResNet
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(Resnet_model.parameters(), lr= 0.01)

    # 三、 训练
    Iter_Time = 3
    loss_train = np.zeros(Iter_Time)
    test_acc= np.zeros(Iter_Time)
    train_acc = np.zeros(Iter_Time)
    print('start to train' + '*'*20)
    for epoch in range(Iter_Time):
        loss = train(model=Resnet_model, train_X=trainLoader, loss_func=loss_func,optimizer=optimizer)
        loss_train[epoch] = loss
        C = predict(model=Resnet_model, test_X = trainLoader)
        train_acc[epoch] = C.trace() / C.sum()
        print("epoch {},训练集准确率:{}".format(epoch,C.trace() / C.sum()))
        print(C)
        print('次数',epoch,'test 分界线','*' * 20)
        C = predict(model=Resnet_model, test_X =testLoader)
        test_acc[epoch] = C.trace() / C.sum()
        print("epoch {},测试集准确率:{}".format(epoch,C.trace() / C.sum()))
        print(C)


    # torchvision.models.AlexNet
    torchvision.models
    # import torch
    # model = torch.load('F:\\python_AI\\cv\\resnet18-5c106cde.pth')
    # print(model)
    # a  = torch.randn(64,3,224,224)







