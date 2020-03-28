# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:03:08 2020

@author: hawki
"""

import torch as tch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1) 
        
    def forward(self,x):
        return self.linear(x)    

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73,80,75],
                       [93,88,93],
                       [89,91,90],
                       [96,98,100],
                       [73,66,70]]
        self.y_data = [[152],[185],[180],[196],[142]]
    
    def __len__(self):
      # 이 데이터셋의 총 개수
        return len(self.x_data)
    
    def __getitem__(self,idx):
      # 어떠한 인덱스를 받았을 때 그에 상응하는 입출력 데이터 반환 후
      # float tensor 형태로 반환
      
        x = tch.FloatTensor(self.x_data[idx])
        y = tch.FloatTensor(self.y_data[idx])
        return x,y

dataset = CustomDataset()
model = MultivariateLinearRegressionModel()
dataloader = DataLoader(
        dataset,
        batch_size=2,
          #  minibatch의 크기 
          #  2의 제곱수로 설정
          #  (16,32,64,...)
        shuffle=True
          #  매번 data가 학습되는 순서 바꿈
          #  dataset의 순서 기억을 방지
        )

opt = optim.SGD(model.parameters(),lr=1)

epochs = 20
for i in range(epochs+1):
    for batch_idx,samples in enumerate(dataloader):
          #  minibatch의 인덱스와 데이터 받음
        x_trn, y_trn = samples
          # h(x) 계산
        pdt = model(x_trn)
        cost = F.mse_loss(pdt,y_trn)
        
        opt.zero_grad()
        cost.backward()
        opt.step()
    