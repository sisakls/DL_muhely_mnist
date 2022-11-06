import torch
import torchvision
import torchvision.transforms as tfs
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets

import data
import model

n_epochs = 1
#batch_size_train = 64
#batch_size_test = 1000
#learning_rate = 0.01
#momentum = 0.5
#log_interval = 10

train_loader, test_loader = data.prep_data()

network = model.Net()
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(2)]

model.test(network, test_loader, test_losses, test_counter)
for epoch in range(1, n_epochs + 1):
    model.train(network, train_loader, train_losses, train_counter, epoch)
    model.test(network, test_loader, test_losses, test_counter)