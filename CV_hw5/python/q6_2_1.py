import torch.nn.init as init

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='../data/oxford-flowers17/train')
parser.add_argument('--val_dir', default='../data/oxford-flowers17/val')
parser.add_argument('--test_dir', default='../data/oxford-flowers17/test')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=15, type=int)
parser.add_argument('--num_epochs2', default=20, type=int)
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--learning_rate1', default = 0.005, type= float)
parser.add_argument('--learning_rate2', default = 0.0001, type= float)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main(args):
  # Figure out the datatype we will use; this will determine whether we run on
  # CPU or on GPU. Run on GPU by adding the command-line flag --use_gpu
  dtype = torch.FloatTensor
  if args.use_gpu:
    dtype = torch.cuda.FloatTensor


  train_transform = T.Compose([
    T.Scale(256),
    T.RandomSizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])

  train_dset = ImageFolder(args.train_dir, transform=train_transform)
  train_loader = DataLoader(train_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True)

  val_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  val_dset = ImageFolder(args.val_dir, transform=val_transform)
  val_loader = DataLoader(val_dset,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)
  test_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  test_dset = ImageFolder(args.test_dir, transform=test_transform)
  test_loader = DataLoader(test_dset,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

  model = torchvision.models.squeezenet1_1(pretrained=True)
  num_classes = len(train_dset.classes)
  # model.fc = nn.Linear(model.fc.in_features, num_classes)
  model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
  model.num_classes = num_classes


  # training the model.
  model.type(dtype)
  # loss_fn = nn.CrossEntropyLoss().type(dtype)
  #

  for param in model.parameters():
    param.requires_grad = False
  for param in model.classifier.parameters():
    param.requires_grad = True

  # Construct an Optimizer object for updating the last layer only.
  optimizer = torch.optim.Adam(model.classifier.parameters(), lr = args.learning_rate1)

  train_acc_hist = []
  train_loss_hist = []

  # Update only the last layer for a few epochs.
  for epoch in range(args.num_epochs1):
    # Run an epoch over the training data.
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
    train_loss = run_epoch(model, train_loader, optimizer, dtype)

    # Check accuracy on the train and val sets.
    train_acc = check_accuracy(model, train_loader, dtype)
    val_acc = check_accuracy(model, val_loader, dtype)

    train_acc_hist.append(train_acc)
    train_loss_hist.append(train_loss)


    print('Train accuracy: ', train_acc)
    print('Val accuracy: ', val_acc)
    print()

  # Now we want to finetune the entire model for a few epochs. To do thise we
  # will need to compute gradients with respect to all model parameters, so
  # we flag all parameters as requiring gradients.
  for param in model.parameters():
    param.requires_grad = True

  # Construct a new Optimizer that will update all model parameters. Note the
  # small learning rate.
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate2)

  # Train the entire model for a few more epochs, checking accuracy on the
  # train and validation sets after each epoch.
  for epoch in range(args.num_epochs2):
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
    train_loss = run_epoch(model, train_loader, optimizer, dtype)

    train_acc = check_accuracy(model, train_loader, dtype)
    val_acc = check_accuracy(model, val_loader, dtype)

    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    print('Train accuracy: ', train_acc)
    print('Val accuracy: ', val_acc)
    print()



  num = np.arange(1, args.num_epochs1 + args.num_epochs2 + 1)
  plt.figure()
  plt.subplot(211)
  plt.title('Accuracy')
  plt.xlabel('iteration')
  plt.ylabel('avg Acc')
  plt.plot(num, train_acc_hist, color = 'g')

  # plt.figure('Loss')
  plt.subplot(212)
  plt.title('Loss')
  plt.xlabel('iteration')
  plt.ylabel('loss')
  plt.plot(num, train_loss_hist, color = 'g')
  plt.show()


def run_epoch(model, loader, optimizer, dtype):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
  model.train()
  total_loss = 0
  for x, y in loader:

    xb = Variable(x.type(dtype))
    yb = Variable(y.type(dtype).long())

    # Run the model forward to compute scores and loss.
    y_pred = model(xb)
    loss = nn.functional.cross_entropy(y_pred, yb)
    total_loss += loss

    # Run the model backward and take a step using the optimizer.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return total_loss


def check_accuracy(model, loader, dtype):
  """
  Check the accuracy of the model.
  """
  # Set the model to eval mode
  model.eval()
  num_correct, num_samples = 0, 0
  for x, y in loader:
    xb = Variable(x.type(dtype), volatile=True)
    y_pred = model(xb)
    _, preds = y_pred.data.cpu().max(1)
    num_correct += (preds == y).sum()
    num_samples += x.size(0)

  # Return the fraction of datapoints that were correctly classified.
  acc = float(num_correct) / num_samples
  return acc


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)


