import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from nn import *
from util import *
from collections import Counter
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# pick a batch size, learning rate
hidden_size = 64
batch_size = 64
train_x = np.array([x.reshape((32,32)) for x in train_x])
test_x = np.array([x.reshape((32,32)) for x in test_x])
print(train_x.shape)

train_x = torch.from_numpy(train_x).type(torch.float32).unsqueeze(1)
train_y = torch.from_numpy(train_y).type(torch.float32)

train_batch = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

params = Counter()

device = torch.device('cpu')

class Net(nn.Module):

    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        # self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)
        # self.conv2 = nn.Conv2d(5, 20, 3)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(14 * 14 * 20, D_in)
        # self.fc2 = nn.Linear(D_in, D_out)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3,1,1),nn.ReLU(),nn.Conv2d(8, 16, 3,1,1),nn.ReLU(),nn.MaxPool2d(2,2))
        self.fc1 = nn.Sequential(nn.Linear(16*16*16, 1024),nn.ReLU(),nn.Linear(1024, 36))

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = x.view(-1, 14 * 14 * 20)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return x
        x = self.conv1(x)
        x = x.view(-1, 16*16*16)
        x = self.fc1(x)
        return x


D_in = train_x.shape[1]
H = hidden_size
D_out = train_y.shape[1]

model = Net(D_in, H, D_out)
criterion = torch.nn.functional.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

lost_hist = []
acc_hist = []
for t in range(50):
    total_loss = 0
    avg_acc = 0
    for train in train_batch:


        xb = train[0]
        yb = train[1]
        true = torch.argmax(yb, 1)

        y_pred = model(xb)

        loss = criterion(y_pred, true)
        total_loss += loss.item()

        predicted = torch.argmax(y_pred, 1)
        avg_acc += torch.sum(predicted == true).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_acc = avg_acc / train_y.shape[0]
    lost_hist.append(total_loss)
    acc_hist.append(avg_acc)

    if t % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(t, total_loss, avg_acc))



num = np.arange(1, 50 + 1)
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



