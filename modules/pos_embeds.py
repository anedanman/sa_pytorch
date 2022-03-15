import torch
from torch import nn

from utils import build_grid


class PosEmbeds(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.linear = nn.Linear(4, hidden_size)
        self.grid = torch.Tensor(build_grid(resolution))
        
    def forward(self, inputs):
        pos_emb = self.linear(self.grid).moveaxis(3, 1)
        return inputs + pos_emb
