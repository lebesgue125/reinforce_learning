import torch as T

dev = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


def numpy_to_tensor(narray, dtype):
    if dtype is 'float':
        return T.from_numpy(narray).float().to(dev)
    if dtype is 'int':
        return T.from_numpy(narray).int().to(dev)
    if dtype is 'long':
        return T.from_numpy(narray).long().to(dev)
    return narray
