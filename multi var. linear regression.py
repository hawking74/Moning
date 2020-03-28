# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:47:31 2020

@author: hawki
"""

import numpy as np
import torch

xtrn = torch.FloatTensor([[73,80,75],
                          [93,88,93],
                          [89,91,90],
                          [96,98,100],
                          [73,66,70]])
ytrn = torch.FloatTensor([[152],[185],[180],[196],[142]])

w = torch.zeros((3,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

print(w)
print(b)

opt = torch.optim.SGD([w,b],lr=1e-5)

epochs = 20
for i in range(1,epochs+1):
    hyp = xtrn.matmul(w)+b
    cost = torch.mean((hyp-ytrn)**2)
    opt.zero_grad()
    cost.backward()
    opt.step()
    
print('print w:',float(w[0]))
print(b)
