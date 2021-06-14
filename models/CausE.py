import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_


class CausE(nn.Module):
    def __init__(self, num_users, num_items, embedding_size,
                 reg_c, reg_t, reg_tc, s_c, s_t, device='cpu'):
        super(CausE, self).__init__()
        self.user_e = nn.Embedding(num_users, embedding_size)
        self.item_e_c = nn.Embedding(num_items, embedding_size)
        self.item_e_t = nn.Embedding(num_items, embedding_size)
        self.user_b = nn.Embedding(num_users, 1)
        self.item_b = nn.Embedding(num_items, 1)
        self.reg_c = reg_c
        self.reg_t = reg_t
        self.reg_tc = reg_tc
        self.s_c = s_c
        self.s_t = s_t

        self.apply(self._init_weights)

        self.loss_c = nn.MSELoss()
        self.loss_t = nn.MSELoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)

    def forward(self, user, item):
        user_embedding = self.user_e(user)
        item_embedding = self.item_e_c(item)

        preds = self.user_b(user)
        preds += self.item_b(item)
        preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        return preds.squeeze()

    def calculate_loss(self, user_list, item_list, label_list, control):
        user_embedding = self.user_e(user_list)

        item_embedding_c = self.item_e_c(item_list)
        item_embedding_t = self.item_e_t(item_list)

        dot_c = (user_embedding * item_embedding_c).sum(dim=1, keepdim=True)
        pred_c = dot_c + self.user_b(user_list) + self.item_b(item_list)
        pred_c = pred_c.squeeze()
        dot_t = (user_embedding * item_embedding_t).sum(dim=1, keepdim=True)
        pred_t = dot_t + self.user_b(user_list) + self.item_b(item_list)
        pred_t = pred_t.squeeze()

        loss = self.loss_c(pred_c, label_list)
        loss += self.loss_t(pred_t, label_list)
        loss_reg_tc = self.reg_tc * torch.norm(item_embedding_c - item_embedding_t, 2)
        return loss + loss_reg_tc

    def predict(self, user, item):
        return self.forward(user, item)

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
