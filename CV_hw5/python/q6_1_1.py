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
batch_size = 32  # random guess
hidden_size = 64

train_x = torch.from_numpy(train_x).type(torch.float32)
train_y = torch.from_numpy(train_y).type(torch.float32)

test_x = torch.from_numpy(test_x).type(torch.float32)
test_y = torch.from_numpy(test_y).type(torch.float32)


params = Counter()

device = torch.device('cpu')

class Net(nn.Module):

    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(D_in, H)  # 6*6 from image dimension
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = torch.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)
        return x


D_in = train_x.shape[1]
H = hidden_size
D_out = train_y.shape[1]

model = Net(D_in, H, D_out)
# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
# loss_fn = torch.nn.MSELoss(reduction='sum')
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

lost_hist = []
acc_hist = []
for t in range(500):
    total_loss = 0
    avg_acc = 0

    y_pred = model(train_x)

    loss = criterion(y_pred, train_y)
    total_loss += loss.item()

    predicted = torch.argmax(y_pred, 1)
    true = torch.argmax(train_y,1)

    avg_acc += torch.sum(predicted == true).item()
    avg_acc = avg_acc / train_y.shape[0]


    lost_hist.append(total_loss)
    acc_hist.append(avg_acc)

    if t % 10 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(t, total_loss, avg_acc))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

num = np.arange(1, 500 + 1)
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
