import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms

# ---------------  import our work ----------------------
from ResNet import *
from InceptionAndSE_block import *



# ---------------- 训练用 train & predict 函数
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
                if (i + 1) % 100 == 0:
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
            x.to(device)
            y.to(device)
            output = model(x)
            if i == 0:
                pred_y = torch.argmax(output, axis=1)
                true_y = y
            else:
                pred_y = torch.cat( (pred_y,torch.argmax(output, axis=1) ) , dim=-1)
                true_y = torch.cat( (true_y , y), dim=-1)
    C = confusion_matrix( y_pred = np.array(pred_y) , y_true= np.array(true_y)  )
    return C


if __name__ == '__main__':
    # 一、 读入数据 与 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize( mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])

    transform_test = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dat = torchvision.datasets.CIFAR10(root='F:\python_AI\cv\cifar-10-python', train=True,download=False ,transform=transform)
    trainLoader = data.DataLoader(train_dat, batch_size=32,shuffle=True)
    test_dat = torchvision.datasets.CIFAR10(root='F:\python_AI\cv\cifar-10-python', train=False,download=False ,transform= transform_test )
    testLoader = data.DataLoader(test_dat, batch_size=32,drop_last=False)


    # 二、 设置模型参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # model = ResNet(block=BasicBlock , layers=[3,3,3], num_classes=10) # ResNet 6*1 + 2

    # model = SE_ResNet(block=SE_BasicBlock , layers=[1,1,1], num_classes=10) # SE-ResNet
    # model = SE_ResNet(block=SE_Bottleneck , layers=[1,1,1], num_classes=10) # SE-ResNet


    # MyInceptionNet()
    model = ResNet(block=BasicBlock , layers=[3,3,3], num_classes=10)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 不特殊说明，我们第二部分的试验



    # 三、 训练
    Iter_Time = 50
    loss_train = np.zeros(Iter_Time)
    test_acc= np.zeros(Iter_Time)
    train_acc = np.zeros(Iter_Time)
    print('start to train' + '*'*20)
    for epoch in range(Iter_Time):
        loss = train(model=model, train_X=trainLoader, loss_func=loss_func,optimizer=optimizer)
        loss_train[epoch] = loss
        C = predict(model=model, test_X = trainLoader)
        train_acc[epoch] = C.trace() / C.sum()
        print("epoch {},训练集准确率:{}".format(epoch,C.trace() / C.sum()))
        # print(C) # 混淆矩阵
        print('训练次数:',epoch, 'test 分界线','*' * 20)
        C = predict(model=model,  test_X =testLoader)
        test_acc[epoch] = C.trace() / C.sum()
        print("epoch {},测试集准确率:{}".format(epoch,C.trace() / C.sum()))

