import torch
from torch.autograd import Variable

def onehot(tensor, n_vals):
    "Convert a (*,c,h,w) (Long)Tensor to one-hot (*,n_vals,h,w) encoding"

    onehot_size = list(tensor.size())
    onehot_size[-3] = n_vals
    onehot = torch.zeros(onehot_size)
    # Fill dimension c with 1's
    # scatter_ expects a LongTensor on cpu
    onehot.scatter_(-3, tensor.cpu().long(), 1)
    return Variable(onehot)
