import torch 
from torch.autograd import Variable 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x))
        #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        #x = F.max_pool2d(F.relu(self.conv2(x)),2)
        #x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print("Model Summary\n")
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = Variable(torch.randn(1,1,32,32))
out = net(input)
print(out)

#loss function 
net.zero_grad()
out.backward(torch.randn(1,10))

output = net(input) 
target = Variable(torch.arange(1,11))
criterion = nn.MSELoss()

loss = criterion(output,target)
print(loss)
print(loss.creator)
print(loss.creator.previous_functions[0][0])
print(loss.creator.previous_functions[0][0].previous_functions[0][0])

#back prop 
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#update weight 
##weight -= learning_rate*gradient 
learning_rate = 0.01 
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)
optimizer = optim.SGD(net.parameters(),0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output,target)
loss.backward()
optimizer.step() #update

