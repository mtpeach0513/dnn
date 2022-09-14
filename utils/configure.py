import torch

DEBUG = False

class Config:
    seed = 1
    lr = 1e-2
    min_lr = 1e-8
    num_epochs = 100
    batch_size = 512
    num_workers = 0
    pin_memory = True
    n_folds = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
