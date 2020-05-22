import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle

#Read in train, valid, test data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']


max_iters = 80
# pick a batch size, learning rate
batch_size = 25  # random guess
learning_rate = 0.006  # random guess
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################

initialize_weights(train_x.shape[1], hidden_size,params,'layer1')
initialize_weights(hidden_size, train_y.shape[1],params,'output')
assert(params['Wlayer1'].shape == (1024,64))
assert(params['blayer1'].shape == (64,))

#visualize original weight after initialization
# fig = plt.figure()
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(8, 8),  # creates 2x2 grid of axes
#                  )
#
# for i in range(hidden_size):
#     grid[i].imshow(params['Wlayer1'][:,i].reshape((32,32)))  # The AxesGrid object work as a list of axes.
#     plt.axis('off')
# plt.show()

# store the history values for use of the plot later
acc_hist_train = []
lost_hist_train = []
acc_hist_valid = []
lost_hist_valid = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_loss_valid = 0
    avg_acc = 0
    avg_acc_valid = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        h1 = forward(xb, params,'layer1')
        probs = forward(h1,params,'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        avg_acc += acc

        # backward
        delta1 = probs - yb
        # delta1[np.arange(probs.shape[0]),y_idx] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']


        ##########################
        ##### your code here #####
        ##########################
    avg_acc = avg_acc / batch_num
    acc_hist_train.append(avg_acc)
    lost_hist_train.append(total_loss)


    #running on valid data
    h1 = forward(valid_x, params,'layer1')
    probs = forward(h1,params,'output',softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    avg_acc_valid = acc
    total_loss_valid += loss
    acc_hist_valid.append(avg_acc_valid)
    lost_hist_valid.append(total_loss_valid)



    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))


##########################
##### your code here #####
##########################
#plot accuracy & loss

# num = np.arange(1, max_iters + 1)
# plt.figure()
# plt.subplot(211)
# plt.title('Accuracy')
# plt.xlabel('iteration')
# plt.ylabel('avg Acc')
# plt.plot(num, acc_hist_train, color = 'g')
# plt.plot(num, acc_hist_valid, color = 'b')
# plt.legend(['train','valid'])
#
# # plt.figure('Loss')
# plt.subplot(212)
# plt.title('Loss')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.plot(num, lost_hist_train, color = 'g')
# plt.plot(num, lost_hist_valid, color = 'b')
# plt.legend(['train','valid'])
# plt.show()



# run on validation set and report accuracy! should be above 75%
h1 = forward(valid_x, params,'layer1')
probs = forward(h1,params,'output',softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
print('Validation accuracy: ',valid_acc)

h1 = forward(test_x, params,'layer1')
probs = forward(h1,params,'output',softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, probs)
print('Test accuracy: ',test_acc)



if False:
    for crop in xb:

        plt.imshow(crop.reshape(32,32).T)
        plt.show()


saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Q3.3to


# visualize weights here
##########################
##### your code here #####
##########################

#visualize layer1 weight after training
# fig = plt.figure()
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(8, 8),  # creates 2x2 grid of axes
#                  )
#
# for i in range(hidden_size):
#     grid[i].imshow(params['Wlayer1'][:,i].reshape((32,32)))  # The AxesGrid object work as a list of axes.
#     plt.axis('off')
# plt.show()



# Q3.4
true_y = np.argmax(test_y, axis = 1)
pred_y = np.argmax(probs, axis = 1)
# compute comfusion matrix here
##########################
##### your code here #####
##########################
confu_mat = confusion_matrix(true_y, pred_y)

# import string
# plt.imshow(confu_mat,interpolation='nearest')
# plt.grid(True)
# plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.show()
