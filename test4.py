import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.nn import Parameter
import sys
sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs
from tasp.MarginalObjectives import elbo
from torch.autograd.gradcheck import zero_gradients
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler

class Encoder(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.lin1 = nn.Linear(1, 2)
        self.lin2 = nn.Linear(2, 4)
        self.lin3 = nn.Linear(4, 1)
        self.lin4 = nn.Linear(2, 1)

    def forward(self,data,IsTrue):
        if IsTrue:
            x = self.lin1(data)
            x = self.lin2(x)
            x = self.lin3(x)
        else:
            x = self.lin1(data)
            x = self.lin4(x)
        return x


# print('lin1.weight',lin1.weight)
# print(lin2.weight)
# print(lin3.weight)
# print(lin4.weight)

# loss.backward(retain_graph=True)
# optimizer.step()

# print('lin1.weight',lin1.weight)
# print(lin2.weight)
# print(lin3.weight)
# print(lin4.weight)
def func(IsSum):
    model = Encoder()
    model.train()
    with torch.no_grad():
        model.lin1.weight[0, 0] = 2.
        model.lin1.weight[1, 0] = 2.
        model.lin2.weight[0, 0] = 3.
        model.lin2.weight[1, 0] = 3.
        model.lin2.weight[2, 0] = 3.
        model.lin2.weight[3, 0] = 3.
        model.lin2.weight[0, 1] = 3.
        model.lin2.weight[1, 1] = 3.
        model.lin2.weight[2, 1] = 3.
        model.lin2.weight[3, 1] = 3.
        model.lin3.weight[0, 0] = 4.
        model.lin3.weight[0, 1] = 4.
        model.lin3.weight[0, 3] = 4.
        model.lin3.weight[0, 3] = 4.
        model.lin4.weight[0, 0] = 5.
        model.lin4.weight[0, 1] = 5.

    data_gt = torch.tensor([7.0])
    data2_gt = torch.tensor([7.0])
    lr = 1e-1  # learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    data = torch.tensor([3.0], requires_grad=True)
    output = model(data, True)
    loss1 = torch.pow(output - data_gt, 2)

    data2 = torch.tensor([2.0], requires_grad=True)
    output2 = model(data2, False)
    loss2 = torch.pow(output2 - data2_gt, 2)
    if IsSum:
        loss = loss1+ loss2
        loss.backward(retain_graph=True)
        optimizer.step()
        pass
    else:

        # loss = loss1 + loss2
        # print('lin1.weight',lin1.weight)
        # print(lin2.weight)
        # print(lin3.weight)
        # print(lin4.weight)

        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        optimizer.step()

    return model.lin1.weight

print(func(False))
print(func(True))

