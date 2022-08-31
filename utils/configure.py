import torch

DEBUG = False

class Config:
    seed = 1
    lr = 1e-4
    max_lr = 5e-3
    num_epochs = 40
    batch_size = 512
    num_workers = 0
    pin_memory = False
    n_folds = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
