import torch 
from torch.autograd import Variable 

#variable
x = Variable(torch.ones(2,2), requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.creator)
print(y.data)

z = y*y*3
out = z.mean()
print(x, y, z, out)
#print(z, out)

#gradient
out.backward()
print(x.grad)
print(z.grad)

#practice 
x = torch.randn(3)
x = Variable(x,requires_grad=True)

y = x*2
while y.data.norm() < 1000:
    y = y*2
print(y)

gradient = torch.FloatTensor([1000,1.0,0.001])
y.backward(gradient)

print(x.grad)

