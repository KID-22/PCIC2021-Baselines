import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
from .loss import IPSLoss


class MF_IPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, inverse_propensity, device):
        super(MF_IPS, self).__init__()
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.inverse_propensity = inverse_propensity

        self.user_e = nn.Embedding(self.num_users, embedding_size)
        self.item_e = nn.Embedding(self.num_items, embedding_size)
        self.user_b = nn.Embedding(self.num_users, 1)
        self.item_b = nn.Embedding(self.num_items, 1)

        self.apply(self._init_weights)

        self.loss = IPSLoss(device)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)

    def forward(self, user, item):
        user_embedding = self.user_e(user)
        item_embedding = self.item_e(item)

        preds = self.user_b(user)
        preds += self.item_b(item)
        preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)

        return preds.squeeze()

    def calculate_loss(self, user_list, item_list, label_list):
        return self.loss(self.forward(user_list, item_list), label_list, self.inverse_propensity, item_list)

    def predict(self, user, item):
        return self.forward(user, item)

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_embedding(self):
        return self.user_e, self.item_e

