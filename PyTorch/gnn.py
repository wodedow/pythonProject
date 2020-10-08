import time
import torch
from torch import nn, optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
