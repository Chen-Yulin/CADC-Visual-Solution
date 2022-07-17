#gpu方式的mnist手写数字识别
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import numpy as np
from torchvision import datasets, models, transforms
EPOCH = 5
BATCH_SIZE = 50
LR = 0.0003
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False
)
#change in here
test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:5000].cuda()/255.   # Tensor on GPU
test_y = test_data.targets[:5000].cuda()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2),
                                   nn.ReLU(), nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120), nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))
        # super(CNN,self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1,16,5,1,2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16,32,5,1,2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        # self.out = nn.Linear(32 * 7 * 7,10) #10分类的问题
 
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def main():
    cnn = CNN()
    cnn.cuda()
 
    optimizer = optim.Adam(cnn.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()
 
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            # print(b_y)
            output = cnn(b_x)
            loss = loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                test_output = cnn(test_x)
 
                # !!!!!!!! Change in here !!!!!!!!! #
                pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # move the computation in GPU
 
                accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
    torch.save(cnn.state_dict(),'model_dict.pth')
if __name__ == '__main__':
    main()
    
    # cnn_predict=CNN()
    # cnn_predict.load_state_dict(torch.load('model_dict.pth'))
    # cnn_predict.eval()
    # img=cv2.imread('num_img.jpg',cv2.IMREAD_GRAYSCALE)
    # print(img)
    # img = Image.fromarray(img)
    # transf = transforms.ToTensor()
    # img=transf(img)
    # img = img.view(1, 1, 28, 28)
    # test_output = cnn_predict(img)
    # _, prediction = torch.max(test_output, 1)
    # print(prediction)