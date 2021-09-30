import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, input_size, n_cats, n_neurons=[100, 100, 100], drate=0):
        super(Net, self).__init__()
        self.inputsize = input_size
        self.func = nn.ModuleList()


        for n in range(len(n_neurons) - 1):
            n_0 = n_neurons[n]
            n_1 = n_neurons[n + 1]
            self.func.append(nn.Linear(n_0, n_1, bias=True))
            self.func.append(nn.ReLU())
            self.func.append(nn.Dropout(drate))

        self.func.append(nn.Linear(n_1, n_cats, bias=True))

    def forward(self, x):
        for f in self.func:
            x = f(x)
        return x

def calc_loss_l1(x1, x2):
    l1 = torch.abs(x1 - x2)
    return l1.mean()
