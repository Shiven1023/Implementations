import torch
import torch.nn as nn
import numpy as np


class LoRA(nn.Module):
    def __init__(self, original, r, alpha):
        super().__init__()
        self.original = original
        self.r = r
        self.alpha = alpha
        self.in_dim = original.in_features
        self.out_dim = original.out_features
        self.A = nn.Parameter(torch.randn(self.in_dim,r))
        self.B = nn.Parameter(torch.zeros(self.r,self.out_dim))
    def forward(self , x):
        h = self.original(x) + (self.alpha/self.r)*( x @ (self.A @ self.B))
        return h
linear = nn.Linear(512, 1536)
linear.weight.requires_grad = False
linear.bias.requires_grad = False

lora = LoRA(linear, r=8, alpha=16)
x = torch.randn(10, 512)
print(lora(x).shape)  # should be (10, 1536)

for name, p in lora.named_parameters():
    print(name, p.requires_grad)