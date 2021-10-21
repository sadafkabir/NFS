import torch
import numpy as np
import torch.optim as optim
from helper.dataloader import Dataloader
from helper.network import Net
from helper.train import train
from helper.train import compute_grad
from helper.test import test


def main(n_neurons, batch_size, drate, fold, display=False, seed=123):
    # construct the model
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    dl = Dataloader(file_address='./dermatology.mat', batch_size=batch_size, selected_fold=fold)

    n_features = dl.n_features
    n_neurons[0] = n_features

    ae = Net(input_size=n_features, n_neurons=n_neurons, drate=drate, n_cats=dl.n_cats)
    # set model to train mode
    ae.train()

    # define optimizer
    optimizer = optim.Adam(ae.parameters(), lr=1e-3, weight_decay=0e-5)
    train_epoch = 1000

    best_acc = 0
    for epoch in range(train_epoch):
        train(ae, epoch, dl, optimizer, batch_size, display=display)
        acc = test(ae, epoch, dl, best_acc, batch_size, display=display)
        if acc > best_acc:
            best_acc = acc
        torch.save(ae.state_dict(), './checkpoint/test0.pth')
    print("best acc: %.6f" % (best_acc))
    scores = compute_grad(ae, epoch, dl, optimizer, batch_size)

    return best_acc, scores



if __name__ == "__main__":
    n_neurons = [30, 30]

    batch_size = 32
    display_steps = 100
    main(n_neurons, batch_size, drate=0.6, fold=0, display=True)

