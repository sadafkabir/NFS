import torch

def test(net, epoch, dl_test, best_acc, batch_size, display):
    net.eval()
    acc = 0
    loss = 0
    total_batches = dl_test.X_test.shape[0] // batch_size
    t_sample = 0
    with torch.no_grad():
        for iter in range(total_batches):
            x, y = dl_test.next_test()

            bs = x.size(0)
            y_pred = net(x+torch.randn_like(x)*2)

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            acc += pred.eq(y.view_as(pred)).type(torch.float).sum().item()
            t_sample += bs
    acc /= t_sample

    if acc > best_acc:
        best_acc = acc
    if display:
        print('\n Epoch: %02d, test results: acc: %.6f, best acc: %.6f \n' % (epoch, acc, best_acc))
    # exit()
    return acc
