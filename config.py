# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    model = 'MF_Naive'
    is_eval_ips = False

    data_dir = './data/'

    train_data = data_dir + '/train/extract_alldata.txt'
    val_all_data = data_dir + '/valid/validation.txt'
    test_data = data_dir + '/test/test_P1.txt'

    # IPS data
    ps_train_data = data_dir + '/train/extract_alldata.txt'
    ps_val_data = data_dir + '/valid/validation.txt'

    # CausE data
    s_c_data = data_dir + '/train/extract_alldata.txt'
    s_t_data = data_dir + '/valid/validation_P1.txt'
    cause_val_data = data_dir + '/valid/validation_P2.txt'

    reg_c = 0.001
    reg_t = 0.001
    reg_tc = 0.001

    metric = 'mse'
    verbose = 50

    device = 'cpu'
    batch_size = 512
    embedding_size = 16

    max_epoch = 50
    lr = 0.001 
    weight_decay = 1e-5

opt = DefaultConfig()
