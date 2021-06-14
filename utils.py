from torch.utils import data
import numpy as np
from metrics import AUC, MAE, MSE, RMSE, MAE_ips, MSE_ips, RMSE_ips
import torch


class MF_DATA(data.Dataset):
    def __init__(self, filename):
        raw_matrix = np.loadtxt(filename)
        self.users_num = int(1000)
        self.items_num = int(1720)
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


class CausE_DATA(data.Dataset):
    def __init__(self, s_c_data, s_t_data):
        raw_matrix_c = np.loadtxt(s_c_data)
        raw_matrix_t = np.loadtxt(s_t_data)
        self.s_c = raw_matrix_c
        self.s_t = raw_matrix_t
        raw_matrix = np.vstack((raw_matrix_c, raw_matrix_t))
        self.users_num = int(1000)
        self.items_num = int(1720)
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def evaluate_model(model, val_data, opt):
    true = val_data[:, 2]
    user = torch.LongTensor(val_data[:, 0]).to(opt.device)
    item = torch.LongTensor(val_data[:, 1]).to(opt.device)
    preds = model.predict(user, item).to(opt.device)

    mae = MAE(preds, true)
    mse = MSE(preds, true)
    rmse = RMSE(preds, true)
    auc = AUC(true, preds.detach().cpu().numpy())

    return mae, mse, rmse, auc


def evaluate_IPS_model(model, val_data, inverse_propensity, opt):
    true = val_data[:, 2]
    user = torch.LongTensor(val_data[:, 0]).to(opt.device)
    user_num = max(user)
    item = torch.LongTensor(val_data[:, 1]).to(opt.device)
    item_num = max(item)
    preds = model.predict(user, item).to(opt.device)

    mae = MAE_ips(preds, true, item, user_num, item_num, inverse_propensity)
    mse = MSE_ips(preds, true, item, user_num, item_num, inverse_propensity)
    rmse = RMSE_ips(preds, true, item, user_num, item_num, inverse_propensity)

    return mae, mse, rmse