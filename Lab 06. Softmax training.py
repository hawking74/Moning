# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 01:30:22 2020

@author: hawki
"""

import torch as tch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

Basic

'''

tch.manual_seed(1)

z = tch.rand(3,5,requires_grad=True)  #make size(3,5) matrix with random float of <!, learn grad
print(z)
hyp = F.softmax(z,dim=1) # dim= 1 에 대해 softmax 실행
print(hyp)

y = tch.randint(5,(3,)).long()  # 예측값에 대한 정답 index random하게 생성
print(y)

y_one_hot = tch.zeros_like(hyp)  #hyp 와 크기가 같은  one hot tensor 제작
y_one_hot.scatter_(1,y.unsqueeze(1),1)  # dim =1 에 대해 y.unsqueeze

print(y_one_hot)

cost = (y_one_hot*-tch.log(hyp)).sum(dim=1).mean() #1
print(cost)

# tch.log(F.softmax(z,dim=1)) == F.log_softmax(z,dim=1)

F.nll_loss(F.log_softmax(z,dim=1),y)  #negative log likelihood #2

F.cross_entropy(z,y) #3

  #  #1,#2,#3 는 동일 방법





'''

Training with low level cross entropy

'''

x_trn = [[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,7,7]]

y_trn = [2,2,2,1,1,1,0,0]
x_trn = tch.FloatTensor(x_trn)
y_trn = tch.LongTensor(y_trn)

w = tch.zeros((4,3),requires_grad=True)  
  #  4개의 값 입력을 3개의 클래스로 분류를 위한 크기 (4,3)의 w
b = tch.zeros(1,requires_grad=True)

opt = optim.SGD((w,b), lr = 0.05)
epochs = 3000
  # 1000으로는 cost function 이 0으로 수렴하지 않기에 epoch 값 증가

for i in range(epochs):
    hyp = F.softmax(x_trn.matmul(w)+b,dim=1)
    y_one_hot = tch.zeros_like(hyp)
    y_one_hot.scatter_(1,y_trn.unsqueeze(1),1)
    cost = sum((y_one_hot*-tch.log(hyp)).sum(dim=1))  # cost를 전체의 합으로 설정 (youtube 와 별도로)
    opt.zero_grad()
    cost.backward()
    opt.step()
    
    if i%100 == 0:
        print('Epoch :',i,'  cost :',cost)
        
        



'''

Training with F.cross_entropy

'''

x_trn = [[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,7,7]]

y_trn = [2,2,2,1,1,1,0,0]
x_trn = tch.FloatTensor(x_trn)
y_trn = tch.LongTensor(y_trn)

w = tch.zeros((4,3),requires_grad=True)
b = tch.zeros(1,requires_grad=True)

opt = optim.SGD((w,b), lr = 0.1)

epochs = 10000

for i in range(epochs+1):
    z = x_trn.matmul(w) + b
    cost = F.cross_entropy(z,y_trn) 
    opt.zero_grad()
    cost.backward()
    opt.step()
    
    



'''

high level implementation with nn.Module

'''

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)  #4 input to 3 class 
    
    def foward(self,x):
        return self.linear(x)
    
model = SoftmaxClassifierModel()

x_trn = [[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,7,7]]

y_trn = [2,2,2,1,1,1,0,0]
x_trn = tch.FloatTensor(x_trn)
y_trn = tch.LongTensor(y_trn)

w = tch.zeros((4,3),requires_grad=True)
b = tch.zeros(1,requires_grad=True)

opt = optim.SGD(model.parameters(), lr = 0.1)

epochs = 1000

for i in range(epochs+1):
    pdt = model(x_trn)
    cost = F.cross_entropy(pdt,y_trn)  # z 와 실제 정답을 비교
    
    opt.zero_grad()
    cost.backward()
    opt.step()
    
    if i%100 == 0:
        print('Epoch :',i,'  cost :',cost)
        
  # 마지막 코드의 에러의 원인은 잘 모르겠습니다...
