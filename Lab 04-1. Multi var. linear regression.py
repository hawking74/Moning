# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:47:31 2020

@author: hawki
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    
    '''
    hypothesis = x1*w1 + x2*w2 + x3*w3 + ... + b
               = x * w 
               = x.matmul(w) + b
    '''
    
    cost = torch.mean((hyp-ytrn)**2)
      # cost function -> MSE 사용
    opt.zero_grad()  # gradient 초기화
    cost.backward()  # gradient 계산
    opt.step()  # 계산된 gradient 의 방향대로 w,b 업데이트
    
    
    
    
    
'''

nn.Module과 torch.nn.functional을 이용한 코드 단순화

'''

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1) 
        # linear(입력 차원, 출력 차원)
        
    def forward(self,x):
        return self.linear(x)  
    
model = MultivariateLinearRegressionModel()
opt = torch.optim.SGD([w,b],lr=1e-5)
epochs = 20
for i in range(epochs+1):
    hyp = model(x_trn)
      # = xtrn.matmul(w)+b
    cost = F.mse_loss(hyp,y_trn)
      #  = torch.mean((hyp-ytrn)**2)
    opt.zero_grad()
    cost.backward()
    opt.step()
    

