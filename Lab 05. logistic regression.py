# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:53:15 2020

@author: hawki
"""

import torch as tch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

tch.manual_seed(1)

x_dat = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_dat = [[0],[0],[0], [1],[1],[1]]

x_trn = tch.FloatTensor(x_dat) #x_dat using as traning
y_trn = tch.FloatTensor(y_dat)

w = tch.zeros((2,1),requires_grad=True)
b = tch.zeros(1,requires_grad=True)

opt = optim.SGD((w,b),lr=1) #w,b,learningrate=1,using SGD for optimization 

epochs = 1000


for i in range(epochs):
    hyp = tch.sigmoid(x_trn.matmul(w)+b) #hyp = 1/(1+tch.exp(-(x_trn.matmul(w)+b)))
    # estimated class by sigmoid
    cost = F.binary_cross_entropy(hyp,y_trn) #cross entropy for binary classification
    opt.zero_grad()  #grad ro 0 초기화
    cost.backward()  #
    opt.step()   # w,b min 되도록 wb update

#Model 평가

hyp = tch.sigmoid(x_trn.matmul(w)+b)
print(hyp[:])

pdt = hyp >= tch.FloatTensor([0.5]) 
  #  if probability > 5: true,
  #  if not : false    
  #  pdt -> byte tensor
print(pdt)
crt_pdt = pdt.float() == y_trn
  #  byte tensor to float
print(crt_pdt)





'''
higher implementation with class
'''

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8,1)  #w,b 포함 linear layer 
        self.sigmoid = nn.Sigmoid()  
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
model = BinaryClassifier()

x_dat = [[-0.2941,  0.4874,  0.1803, -0.2929,  0.0000,  0.0015, -0.5312, -0.0333],
        [-0.8824, -0.1457,  0.0820, -0.4141,  0.0000, -0.2072, -0.7669, -0.6667],
        [-0.0588,  0.8392,  0.0492,  0.0000,  0.0000, -0.3055, -0.4927, -0.6333],
        [-0.8824, -0.1055,  0.0820, -0.5354, -0.7778, -0.1624, -0.9240,  0.0000],
        [ 0.0000,  0.3769, -0.3443, -0.2929, -0.6028,  0.2846,  0.8873, -0.6000]]
y_dat = [[0.],
        [1.],
        [0.],
        [1.],
        [0.]]


x_trn = tch.FloatTensor(x_dat) 
y_trn = tch.FloatTensor(y_dat)

opt = optim.SGD(model.parameters(),lr=1)  # w,b,learningrate=1,using SGD for optimization 
epochs = 100
for i in range(epochs+1):
    hyp = model(x_trn)
    cost = F.binary_cross_entropy(hyp,y_trn)  # binary classification 을 위한 cross entropy 
    opt.zero_grad()  # gradient 0 초기화
    cost.backward()  
    opt.step()   # w,b update
    
    pdt = hyp >= tch.FloatTensor([0,5])  
    crt_pdt = pdt.float()  
    accuracy = crt_pdt.sum().item()/len(crt_pdt)







