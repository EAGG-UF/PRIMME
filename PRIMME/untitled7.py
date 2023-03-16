#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:22:02 2023

@author: joseph.melville
"""


#I want to see if I can get pytorch working at a fundamental level first

#create a pytorch model that models a linear function


import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt




# Data
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)







# Model
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


#Train
inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.01 
epochs = 100

model = linearRegression(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()
    
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


for name, param in model.named_parameters():
    print(name)
    print(param)


# Test
with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()






# x = np.arange(11)
# y = (2*x+1)[:,None]
# o = np.ones(x.shape)
# A = np.stack([o,x]).T

# np.matmul(np.linalg.pinv(A),y)











ng = 64
sz = torch.Tensor([256,256])
R = torch.rand((64,int(torch.prod(sz))))
D = torch.ones(ng, ng)



D = torch.rand(ng, ng)
i,j = torch.triu_indices(ng,ng)
D.T[i,j] = D[i,j]
for i in range(ng): D[i,i] = 0



plt.imshow(D, aspect='auto')
plt.imshow(R, aspect='auto')

N = torch.matmul(D, R)
Na = torch.argmin(N, 0)
Naa = torch.zeros(N.shape)
Naa[Na,torch.arange(len(Na))] = 1



aaa = torch.matmul(Naa, torch.linalg.pinv(R))
bbb = 1-(aaa-torch.min(aaa))/torch.max((aaa-torch.min(aaa)))



plt.imshow(aaa, aspect='auto')
plt.imshow(bbb, aspect='auto')






np.corrcoef(D.flatten(), aaa.flatten())
