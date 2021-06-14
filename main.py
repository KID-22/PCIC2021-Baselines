from config import opt
import os
import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from metrics import AUC
from utils import MF_DATA, CausE_DATA, evaluate_model
import numpy as np
import argparse
import random
import torch
import copy

seed_num = 2021
print("seed_num:", seed_num)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed_num)


# propensity estimation for MF_IPS
def cal_propensity_score():
    ps_train_data = np.loadtxt(opt.ps_train_data)
    ps_train_data = ps_train_data.astype(int)
    ps_val_data = np.loadtxt(opt.ps_val_data)
    ps_val_data = ps_val_data.astype(int)

    user_num = 1000
    item_num = 1720
    P_L_TO = np.bincount(ps_train_data[:, 2], minlength=2)[:]
    tmp = P_L_TO.sum()
    P_L_TO = P_L_TO / P_L_TO.sum()

    P_L_T = np.bincount(ps_val_data[:, 2], minlength=2)[:]
    P_L_T = P_L_T / P_L_T.sum()

    P_O_T = tmp / (user_num * item_num)
    P = P_L_TO * P_O_T / P_L_T

    propensity_score = [P] * item_num

    return propensity_score


# train for CausE
def train_CausE():
    train_data = CausE_DATA(opt.s_c_data, opt.s_t_data)
    val_data = MF_DATA(opt.cause_val_data)
    train_dataloader_s_c = DataLoader(train_data.s_c,
                                      opt.batch_size,
                                      shuffle=True)
    train_dataloader_s_t = DataLoader(train_data.s_t,
                                      opt.batch_size,
                                      shuffle=True)
    model = getattr(models,
                    opt.model)(train_data.users_num, train_data.items_num,
                               opt.embedding_size, opt.reg_c, opt.reg_c,
                               opt.reg_tc, train_data.s_c[:, :2].tolist(),
                               train_data.s_t[:, :2].tolist())

    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

    best_mse = 10000000.
    best_mae = 10000000.
    best_auc = 0
    best_iter = 0

    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in tqdm(enumerate(train_dataloader_s_c)):
            # train model
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(),
                                        item.long(),
                                        label.float(),
                                        control=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % opt.verbose == 0:
            print('Epoch %d :' % (epoch))
            print('s_c Loss = ', loss.item())

        for i, data in tqdm(enumerate(train_dataloader_s_t)):
            # train model
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(),
                                        item.long(),
                                        label.float(),
                                        control=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        (mae, mse, rmse, auc) = evaluate_model(model, val_data, opt)

        if opt.metric == 'mae':
            if mae < best_mae:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mae-model.pth")
        elif opt.metric == 'mse':
            if mse < best_mse:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mse-model.pth")
        elif opt.metric == 'auc':
            if auc > best_auc:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-auc-model.pth")

        if epoch % opt.verbose == 0:
            print('s_t Loss = ', loss.item())
            print(
                'Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f [%.1f s]'
                % (mae, mse, rmse, auc, time() - t1))
            print("------------------------------------------")

    print("train end\nBest Epoch %d:  MAE = %.4f, MSE = %.4f, AUC = %.4f" %
          (best_iter, best_mae, best_mse, best_auc))

    best_model = getattr(models,
                         opt.model)(train_data.users_num, train_data.items_num,
                                    opt.embedding_size, opt.reg_c, opt.reg_c,
                                    opt.reg_tc, train_data.s_c[:, :2].tolist(),
                                    train_data.s_t[:, :2].tolist())
    best_model.to(opt.device)

    if opt.metric == 'mae':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mae-model.pth"))
    elif opt.metric == 'mse':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mse-model.pth"))
    elif opt.metric == 'auc':
        best_model.load_state_dict(torch.load("./checkpoint/ci-auc-model.pth"))

    print("\n========================= best model =========================")
    mae, mse, rmse, auc = evaluate_model(best_model, train_data, opt)
    print('Train MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    mae, mse, rmse, auc = evaluate_model(best_model, val_data, opt)
    print('Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    print("===============================================================\n")

    return best_model


# train for MF_Naive and MF_IPS
def train(propensity_score):
    print('train begin')

    train_all_data = MF_DATA(opt.train_data)
    train_data = copy.deepcopy(train_all_data)
    val_data = MF_DATA(opt.val_all_data)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)

    if opt.model == 'MF_IPS':
        inverse_propensity = np.reciprocal(propensity_score)
        model = getattr(models, opt.model)(train_all_data.users_num,
                                           train_all_data.items_num,
                                           opt.embedding_size,
                                           inverse_propensity, opt.device)
    elif opt.model == 'MF_Naive':
        model = getattr(models, opt.model)(train_all_data.users_num,
                                           train_all_data.items_num,
                                           opt.embedding_size, opt.device)

    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

    best_mse = 10000000.
    best_mae = 10000000.
    best_auc = 0
    best_iter = 0

    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in tqdm(enumerate(train_dataloader)):
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(), item.long(),
                                        label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time()

        (mae, mse, rmse, auc) = evaluate_model(model, val_data, opt)

        if opt.metric == 'mae':
            if mae < best_mae:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mae-model.pth")
        elif opt.metric == 'mse':
            if mse < best_mse:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mse-model.pth")
        elif opt.metric == 'auc':
            if auc > best_auc:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-auc-model.pth")

        if epoch % opt.verbose == 0:
            print('Epoch %d [%.1f s]:', epoch, t2 - t1)
            print('Train Loss = ', loss.item())
            print(
                'Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f [%.1f s]'
                % (mae, mse, rmse, auc, time() - t2))
            print("------------------------------------------")

    print("train end\nBest Epoch %d:  MAE = %.4f, MSE = %.4f, AUC = %.4f" %
          (best_iter, best_mae, best_mse, best_auc))

    if opt.model == 'MF_IPS':
        inverse_propensity = np.reciprocal(propensity_score)
        best_model = getattr(models, opt.model)(train_all_data.users_num,
                                                train_all_data.items_num,
                                                opt.embedding_size,
                                                inverse_propensity, opt.device)
    elif opt.model == 'MF_Naive':
        best_model = getattr(models, opt.model)(train_all_data.users_num,
                                                train_all_data.items_num,
                                                opt.embedding_size, opt.device)

    best_model.to(opt.device)

    if opt.metric == 'mae':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mae-model.pth"))
    elif opt.metric == 'mse':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mse-model.pth"))
    elif opt.metric == 'auc':
        best_model.load_state_dict(torch.load("./checkpoint/ci-auc-model.pth"))

    print("\n========================= best model =========================")
    mae, mse, rmse, auc = evaluate_model(best_model, train_data, opt)
    print('Train MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    mae, mse, rmse, auc = evaluate_model(best_model, val_data, opt)
    print('Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    print("===============================================================\n")

    return best_model


# gengerate submit file
def generate_submit(model):
    test_data = np.loadtxt(opt.test_data, dtype=int)
    user = torch.LongTensor(test_data[:, 0]).to(opt.device)
    item = torch.LongTensor(test_data[:, 1]).to(opt.device)
    pred = model.predict(user, item).to(opt.device)
    pred = pred.detach().cpu().numpy()
    # normalize
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred = pred.reshape(-1, 1)
    submit = np.hstack((test_data, pred))
    np.savetxt("submit.csv", submit, fmt=('%d', '%d', '%f'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--model', default='MF_Naive')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--metric',
                        default='auc',
                        choices=["mae", "mse", "auc"])

    args = parser.parse_args()
    opt.model = args.model
    opt.batch_size = args.batch_size
    opt.max_epoch = args.epoch
    opt.lr = args.lr
    opt.metric = args.metric

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    if opt.model == 'MF_IPS' or opt.model == 'MF_Naive':
        propensity_score = cal_propensity_score()
        propensity_score = np.array(propensity_score).astype(float)
        best_model = train(propensity_score)
        generate_submit(best_model)
    elif opt.model == 'CausE':
        best_model = train_CausE()
        generate_submit(best_model)

    print('end')