import torch
import torch.nn.functional as F
import scipy
import numpy as np


def hungarian_huber_loss(x, y):
    n_objs = x.shape[1]
    pairwise_cost = F.huber_loss(torch.unsqueeze(y, -2).expand(-1, -1, n_objs, -1), torch.unsqueeze(x, -3).expand(-1, n_objs, -1, -1), reduction='none').mean(dim=-1)
    indices = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cost.numpy())))
    transposed_indices = np.transpose(indices, axes=(0, 2, 1))
    final_costs = torch.gather(pairwise_cost, dim=-1, index=torch.LongTensor(transposed_indices))[:, :, 1]
    return final_costs.sum(dim=1).mean()
