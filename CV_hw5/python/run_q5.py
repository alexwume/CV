import numpy as np
import scipy.io
from nn import *
from util import *
from collections import Counter
import matplotlib.pyplot as plt
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')


# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']


# print('train_x shape',train_x.shape)
max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate = 3e-5  #3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################

initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'layer2')
initialize_weights(hidden_size, train_x.shape[1], params, 'output')

#loss and acc
def compute_loss_acc(y, probs):
    # loss, acc = None, None
    acc = None
    loss =  np.sum((y - probs) ** 2)
    # pred_y =  np.argmax(probs, axis=1).reshape((-1,1)) # find the most possible class
    # y_label = np.argmax(y, axis = 1).reshape((-1,1))

    # acc = np.sum(y_label == pred_y)/y.shape[0]


    ##########################
    ##### your code here #####
    ##########################

    return loss, acc

# should look like your previous training loops
lost_hist_train = []
for itr in range(max_iters):
    total_loss = 0
    # avg_acc = 0
    for xb,_ in batches:

        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'layer2', relu)
        probs = forward(h2, params, 'output', sigmoid)

        loss, acc = compute_loss_acc(xb, probs)
        total_loss += loss
        # avg_acc += acc

        delta1 = - 2 * (xb - probs)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'layer2', relu_deriv)
        backwards(delta3, params, 'layer1', relu_deriv)

        params['M_Wlayer1'] = 0.9 * params['M_Wlayer1'] - learning_rate * params['grad_Wlayer1']
        params['M_blayer1'] = 0.9 * params['M_blayer1'] - learning_rate * params['grad_blayer1']
        params['M_Wlayer2'] = 0.9 * params['M_Wlayer2'] - learning_rate * params['grad_Wlayer2']
        params['M_blayer2'] = 0.9 * params['M_blayer2'] - learning_rate * params['grad_blayer2']
        params['M_Woutput'] = 0.9 * params['M_Woutput'] - learning_rate * params['grad_Woutput']
        params['M_boutput'] = 0.9 * params['M_boutput'] - learning_rate * params['grad_boutput']

        params['Wlayer1'] += params['M_Wlayer1']
        params['blayer1'] += params['M_blayer1']
        params['Wlayer2'] += params['M_Wlayer2']
        params['blayer2'] += params['M_blayer2']
        params['Woutput'] += params['M_Woutput']
        params['boutput'] += params['M_boutput']

    lost_hist_train.append(total_loss)
        ##########################
        ##### your code here #####
        ##########################

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot the loss over iterations
num = np.arange(1, max_iters + 1)
plt.figure()
plt.title('Loss')
plt.xlabel('iteration')
plt.ylabel('Total loss')
plt.plot(num, lost_hist_train, color = 'g')
plt.show()

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



# Q5.3.1

# visualize some results
##########################
##### your code here #####
##########################
h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'layer2', relu)
probs = forward(h2, params, 'output', sigmoid)
# print(valid_x.shape)
# inds = [0,50,100,150,200,250,300, 350,400,450]  # A,B,C,D,E
# for i in inds:
#     plt.subplot(2,1,1)
#     plt.imshow(valid_x[i].reshape(32,32).T)
#     plt.subplot(2,1,2)
#     plt.imshow(probs[i].reshape(32,32).T)
#     plt.show()


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
# inds = [0,50,100,150,200,250,300, 350,400,450]  # A,B,C,D,E
psn_val = 0
for i in range (valid_x.shape[0]):
    psn_val += psnr(valid_x[i].reshape(32,32).T, probs[i].reshape(32,32).T)

psn_val = psn_val / valid_x.shape[0]
print('Average PSN Value', psn_val)
