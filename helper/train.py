import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

def train(net, epoch, dl, optimizer, batch_size, display):
    net.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    running_acc = 0

    for iter in range(dl.X_train.shape[0] // batch_size):
        # sleep(0.001)
        optimizer.zero_grad()
        x, y = dl.next_train()
        bs = x.size(0)

        y_pred = net(x)

        loss = criterion(y_pred, y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        running_acc += pred.eq(y.view_as(pred)).type(torch.float).mean().item()

        display_steps = 100
        if iter % display_steps == 0 and iter > 0 and display:
            print('epoch: %2d, iter: %2d/%2d, loss: %.5f, acc: %.4f' % (
                epoch, iter, dl.data.shape[0] // batch_size, running_loss / (iter + 1),
                running_acc / (iter + 1)))


def compute_grad(net, epoch, dl, optimizer, batch_size):
    net.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    running_loss = 0
    running_acc = 0

    total_grad = 0
    for iter in range(dl.X_train.shape[0] // batch_size):
        # sleep(0.001)
        optimizer.zero_grad()
        x, y = dl.next_train()

        bs = x.size(0)

        var = Variable(x, requires_grad=True)

        y_pred = net(var)

        loss = criterion(y_pred, y.view(-1))

        grad = torch.autograd.grad(loss, var)[0]
        grad_np = grad.cpu().detach().numpy()
        total_grad += np.abs(grad_np).mean(0)

    x = np.arange(0, dl.n_features)
    x = x+1
    normalized_grad = total_grad / total_grad.sum()
    print(normalized_grad)
    sorted_grad = np.argsort(normalized_grad)
    print(sorted_grad)
    plt.bar(x, normalized_grad)
    plt.xticks(x)
    plt.title('Dermatology Dataset')
    plt.xlabel('Feature')
    plt.ylabel('Normalized Importance Score')
    plt.show()
    return normalized_grad