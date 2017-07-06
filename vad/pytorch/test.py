import torch 

x = torch.randn(4,4)
print(x.size())
y = x.view(16)
print(y.size())
z = x.view(-1,8)
print(z.size())
