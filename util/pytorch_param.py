import torch as T

dev = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
