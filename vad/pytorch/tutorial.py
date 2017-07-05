from __future__ import print_function 
import torch 
import numpy as np 

#create tensor
x = torch.Tensor(5,3)
print(x)

x = torch.rand(5,3)
print(x)

#addition 
y = torch.randn(5,3)
print(x.size())

result = torch.Tensor(5,3)
torch.add(x,y,out=result)

print(x+y)
print(torch.add(x,y))
print(result)

#index 
print(x)
print(x[:,1])
print(x[0,:])

#numpy bridge 
a = torch.ones(1,5)
b = a.numpy()
print(a)
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

#GPU 
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y


