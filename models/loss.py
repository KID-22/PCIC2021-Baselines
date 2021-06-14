import torch
import torch.nn as nn
import numpy as np


class IPSLoss(nn.Module):
    # use propensity score to debias
    def __init__(self, device):
        super(IPSLoss, self).__init__()
        self.loss = 0.
        self.device = device

    def forward(self, output, label, inverse_propensity, item):
        self.loss = torch.tensor(0.0)
        label0 = label.cpu().numpy() 
        item_list = item.cpu().numpy()

        weight = torch.Tensor(
            list(map(lambda x: (inverse_propensity[item_list[x]][int(label0[x])]),
                     range(0, len(label0))))).to(self.device)
        weightedloss = torch.pow(output - label, 2) * weight
        self.loss = torch.sum(weightedloss)

        return self.loss
