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
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

data_dir = '/home/alex/PycharmProjects/CV_hw1/hw1/data/'
# data_dir = '../data/'  #change to this if the data folder is in the same root location
# train_files_path = open(join(data_dir, 'train_files.txt')).read().splitlines()
# train_x = np.zeros((len(train_files_path), 3, 100, 100))
# for itr,img_path in enumerate (train_files_path):
#     img_path = join(data_dir, img_path)
#     img = Image.open(img_path)
#     img = np.array(img).astype(np.float32)/255
#     img = skimage.transform.resize(img, (100, 100, 3)).T
#     train_x[itr,:,:,:] = img
# np.save('train_x', train_x)
# test_files_path = open(join(data_dir, 'test_files.txt')).read().splitlines()
# test_x = np.zeros((len(test_files_path), 3, 100, 100))
# for itr,img_path in enumerate (test_files_path):
#     img_path = join(data_dir, img_path)
#     img = Image.open(img_path)
#     img = np.array(img).astype(np.float32)/255
#     img = skimage.transform.resize(img, (100, 100, 3)).T
#     test_x[itr,:,:,:] = img
# np.save('test_x', test_x)
train_x = np.load('train_x.npy')
test_x = np.load('test_x.npy')

train_y = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
test_y = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
num_class = np.max(train_y)

# pick a batch size, learning rate
hidden_size = 64
batch_size = 100

train_x = torch.from_numpy(train_x).type(torch.float32)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)
test_x = torch.from_numpy(test_x).type(torch.float32)
test_y = torch.from_numpy(test_y).type(torch.LongTensor)


train_batch = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
test_batch = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=True)

params = Counter()
device = torch.device('cpu')
#
class Net(nn.Module):

    def __init__(self, D_out):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20 * 46 * 46, 64)
        # self.fc2 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, D_out)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 46 * 46 * 20)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

D_in = train_x.shape[1]
H = hidden_size
D_out = num_class + 1

model = Net(D_out)
criterion = torch.nn.functional.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

lost_hist = []
acc_hist = []
for t in range(100):
    total_loss = 0
    avg_acc = 0
    for train in train_batch:

        xb = train[0]
        yb = train[1]
        # true = torch.argmax(yb, 1)

        y_pred = model(xb)

        loss = criterion(y_pred, yb)
        total_loss += loss.item()

        predicted = torch.argmax(y_pred, 1)
        avg_acc += torch.sum(predicted == yb).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # avg_acc = avg_acc / train_y.shape[0]
    # lost_hist.append(total_loss)
    # acc_hist.append(avg_acc)

    # if t % 2  == 0:
    #     print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(t, total_loss, avg_acc))
    #


# num = np.arange(1, 50 + 1)
# plt.figure()
# plt.subplot(211)
# plt.title('Accuracy')
# plt.xlabel('iteration')
# plt.ylabel('avg Acc')
# plt.plot(num, acc_hist, color = 'g')
#
# # plt.figure('Loss')
# plt.subplot(212)
# plt.title('Loss')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.plot(num, lost_hist, color = 'g')
# plt.show()


avg_acc = 0
for test in test_batch:
        xb = test[0]
        yb = test[1]

        y_pred = model(xb)

        predicted = torch.argmax(y_pred, 1)
        avg_acc += torch.sum(predicted == yb).item()

avg_acc = avg_acc / test_y.shape[0]


print('Test accuracy: {}'.format(avg_acc))
