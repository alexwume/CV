import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import scipy.io
from nn import *
from os.path import join
from util import *
from collections import Counter
import skimage
import torchvision

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

train_dir = '../data/oxford-flowers17/train'
val_dir = '../data/oxford-flowers17/val'
test_dir = '../data/oxford-flowers17/test'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# pick a batch size, learning rate
hidden_size = 64
batch_size = 50


train_transform = T.Compose([
    T.Scale(256),
    T.RandomSizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

train_dset = ImageFolder(train_dir, transform=train_transform)
train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)

val_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
val_dset = ImageFolder(val_dir, transform=val_transform)
val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True)

test_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
test_dset = ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=True)


num_class = len(train_dset.classes)


# train_x = torch.from_numpy(train_x).type(torch.float32)
# train_y = torch.from_numpy(train_y).type(torch.LongTensor)
# test_x = torch.from_numpy(test_x).type(torch.float32)
# test_y = torch.from_numpy(test_y).type(torch.LongTensor)

#
# train_batch = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
# test_batch = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=True)

params = Counter()
device = torch.device('cpu')
#
class Net(nn.Module):

    def __init__(self, D_out):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 100, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(100 * 25 * 25, 50)
        self.fc2 = nn.Linear(50, D_out)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 100 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# D_in = train_x.shape[1]
# H = hidden_size
D_out = num_class + 1

model = Net(D_out)
criterion = torch.nn.functional.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

lost_hist = []
acc_hist = []
for t in range(100):
    total_loss = 0
    avg_acc = 0
    size = 0
    for xb, yb in train_loader:

        xb = torch.autograd.Variable(xb.type(torch.FloatTensor))
        yb = torch.autograd.Variable(yb.type(torch.FloatTensor).long())

        # print(xb.shape)

        y_pred = model(xb)

        loss = criterion(y_pred, yb)
        total_loss += loss.item()

        predicted = torch.argmax(y_pred, 1)
        avg_acc += torch.sum(predicted == yb).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        size += xb.size(0)
        # print(xb.shape)
        # print(size)

    avg_acc = avg_acc / size
    lost_hist.append(total_loss)
    acc_hist.append(avg_acc)

    if t % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(t, total_loss, avg_acc))



num = np.arange(1, 100 + 1)
plt.figure()
plt.subplot(211)
plt.title('Accuracy')
plt.xlabel('iteration')
plt.ylabel('avg Acc')
plt.plot(num, acc_hist, color = 'g')

# plt.figure('Loss')
plt.subplot(212)
plt.title('Loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.plot(num, lost_hist, color = 'g')
plt.show()

#
# avg_acc = 0
# for test in test_batch:
#         xb = test[0]
#         yb = test[1]
#
#         y_pred = model(xb)
#
#         predicted = torch.argmax(y_pred, 1)
#         avg_acc += torch.sum(predicted == yb).item()
#
# avg_acc = avg_acc / test_y.shape[0]
#
#
# print('Test accuracy: {}'.format(avg_acc))
