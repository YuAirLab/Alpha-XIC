#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   train.py
@Author :   Song
@Time   :   2021/1/11 19:26
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import operator
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils.data
import torch.nn.functional
from numba import jit

import warnings
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

import predifine
from dataloader import Model_Dataset
from Model_GRU import Model_GRU

try:
    profile
except:
    profile = lambda x: x


def my_collate(item):
    xic_l, xic_num, label = zip(*item)

    max_row = max(xic_num)
    xic = [np.vstack([xics, np.zeros((max_row - len(xics), xics.shape[1]))]) for xics in xic_l]

    xic = torch.tensor(xic)
    xic_num = torch.tensor(xic_num)
    label = torch.tensor(label)

    return xic, xic_num, label


def train_one_epoch(trainloader, model, optimizer, loss_fn):
    model.train()
    epoch_loss_fg = 0.
    device = predifine.device
    for batch_idx, (batch_xic, batch_xic_num, batch_labels) in enumerate(trainloader):
        batch_xic = batch_xic.float().to(device)
        batch_xic_num = batch_xic_num.long().to(device)
        batch_labels = batch_labels.long().to(device)

        # forward
        prob = model(batch_xic, batch_xic_num)

        # loss
        batch_loss = loss_fn(prob, batch_labels)

        # back
        optimizer.zero_grad()
        batch_loss.backward()
        # update
        optimizer.step()

        # loss
        epoch_loss_fg += (batch_loss.item() * len(batch_xic))

    epoch_loss_fg = epoch_loss_fg / (len(trainloader.dataset))
    return epoch_loss_fg


def eval_one_epoch(evalloader, model):
    model.eval()
    device = predifine.device
    prob_v = []
    label_v = []
    for batch_idx, (batch_xic, batch_fg_num, batch_labels) in enumerate(evalloader):
        batch_xic = batch_xic.float().to(device)
        batch_fg_num = batch_fg_num.long().to(device)
        batch_labels = batch_labels.long().to(device)

        # forward
        prob = model(batch_xic, batch_fg_num)
        prob = torch.softmax(prob.view(-1, 2), 1)
        probs = prob[:, 1].tolist()

        prob_v.extend(probs)
        label_v.extend(batch_labels.cpu().tolist())
    prob_v = np.array(prob_v)
    return prob_v, label_v


def stack_filter_xic(xics):
    xics_len = list(map(len, xics))
    xics = np.vstack(xics)
    # 滤波
    xics = savgol_filter(xics, 11, 3, axis=1)
    # norm
    with np.errstate(divide='ignore', invalid='ignore'):
        xics = xics / xics.max(axis=1, keepdims=True)
    xics[np.isnan(xics)] = 0

    return xics, xics_len


def train_model(X, y, random_state):
    # split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=1 / 10., random_state=random_state)
    train_xics, train_xics_len = stack_filter_xic(X_train)
    valid_xics, valid_xics_len = stack_filter_xic(X_valid)

    train_dataset = Model_Dataset(train_xics, train_xics_len, y_train, type='train')
    valid_dataset = Model_Dataset(valid_xics, valid_xics_len, y_valid, type='valid')

    # dataloader
    torch.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=512,
                                               num_workers=4,
                                               shuffle=True, collate_fn=my_collate)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=1024,
                                               num_workers=2,
                                               shuffle=False, collate_fn=my_collate)
    # define
    model = Model_GRU().to(predifine.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # epoch 1
    loss_ce = train_one_epoch(train_loader, model, optimizer, loss_fn)

    prob_v, label_v = eval_one_epoch(valid_loader, model)
    prob_v[prob_v >= 0.5] = 1
    prob_v[prob_v < 0.5] = 0
    acc_now = sum(prob_v == label_v) / len(label_v)

    print(f'Epoch[{0}], loss: {loss_ce}, acc: {acc_now}')

    return model


@profile
def utils_model(df_pypro, xics_bank, model):
    prob_v = []
    for _, df_batch in df_pypro.groupby(df_pypro.index // 100000):
        print('\rtest numbers of XICs: {}/{} finished'.format(df_batch.index[-1] + 1, len(df_pypro)), end='',
              flush=True)

        # xic
        idx = df_batch['xic_idx'].values
        X = xics_bank[idx]

        # preprocess
        xics, xics_len = stack_filter_xic(X)
        y = np.zeros(len(xics_len))
        dataset = Model_Dataset(xics, xics_len, y, type='test')

        # dataloader
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=2048,
                                                  num_workers=4,
                                                  shuffle=False, collate_fn=my_collate)

        probs, _ = eval_one_epoch(data_loader, model)
        prob_v.extend(probs)
    print('\r')
    return np.array(prob_v)


@profile
def extract_diann_xics(df_input, mzml):
    pd.options.mode.chained_assignment = None

    xics_v = []
    for _, df in df_input.groupby(df_input.index // 100000):
        print('\rextract xics: {}/{} finished'.format(df.index[-1] + 1, len(df_input)), end='', flush=True)
        scans_ms1_rt = mzml.get_ms1_all_rt()
        num_windows = len(mzml.SwathSettings) - 1
        df['query_rt_left'] = df['rt_start'] * 60.
        df['query_rt_right'] = df['rt_stop'] * 60.

        df['idx_start'] = np.abs(df['query_rt_left'].to_numpy().reshape(-1, 1) - scans_ms1_rt).argmin(axis=1) * (
                num_windows + 1)
        df['idx_end'] = np.abs(df['query_rt_right'].to_numpy().reshape(-1, 1) - scans_ms1_rt).argmin(axis=1) * (
                num_windows + 1)
        df['ms2_win_idx'] = np.digitize(df['pr_mz'], mzml.SwathSettings)

        df['xic_num'] = df['query_mz'].apply(len)
        query_mz_v = df['query_mz'].explode().to_numpy().astype(np.float32)

        xic_num_v = df['xic_num'].to_numpy()
        fg_idx_v = [0]
        fg_idx_v.extend(xic_num_v)
        fg_idx_v = np.array(fg_idx_v).cumsum()
        idx_start_v = df['idx_start'].to_numpy()
        idx_end_v = df['idx_end'].to_numpy()
        ms2_win_idx_v = df['ms2_win_idx'].to_numpy()

        for i in range(len(df)):
            fg_idx_start = fg_idx_v[i]
            fg_idx_end = fg_idx_v[i + 1]

            query_mz = query_mz_v[fg_idx_start: fg_idx_end]
            idx_start = idx_start_v[i]
            idx_end = idx_end_v[i]
            ms2_win_idx = ms2_win_idx_v[i]

            xics_ms1, rts_ms1, xics_ms2, rts_ms2 = mzml.get_ms1_ms2_xics_by_lib_mz(idx_start, idx_end,
                                                                                   ms2_win_idx, query_mz,
                                                                                   predifine.MassAccuracyMs1,
                                                                                   predifine.MassAccuracy)
            xics = np.zeros((xic_num_v[i], predifine.target_dim))
            sj_unify_dim(xics_ms1, rts_ms1, xics_ms2, rts_ms2, xics)
            xics = xics[4:, :]
            xics_v.append(xics)

    print('\r')
    xics = np.array(xics_v)
    y = (1 - df_input['decoy']).to_numpy()

    return xics, y


def get_diann_pos_neg_xic(df, xics_bank):
    idx = df['xic_idx'].values
    xics = xics_bank[idx]

    y = (1 - df['decoy']).to_numpy()

    return xics, y


@profile
def train_nn_model(train_loader, nn_model, optimizer, loss_fn):
    nn_model.train()
    device = predifine.device
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.long().to(device)

        # forward
        prob = nn_model(batch_x)
        # loss
        batch_loss = loss_fn(prob, batch_y)
        # back
        optimizer.zero_grad()
        batch_loss.backward()
        # update
        optimizer.step()


@profile
def eval_nn_model(eval_loader, nn_model):
    nn_model.eval()
    loss_v = []
    prob_v = []
    device = predifine.device
    for batch_idx, (batch_x, batch_y) in enumerate(eval_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.long().to(device)

        # forward
        prob = nn_model(batch_x)
        # loss
        batch_loss = torch.nn.functional.cross_entropy(prob, batch_y, reduction='none')
        loss_v.extend(batch_loss.tolist())
        # pred
        prob = torch.softmax(prob.view(-1, 2), 1)
        prob = prob[:, 1].tolist()
        prob_v.extend(prob)

    return 1 - np.array(loss_v).mean(), np.array(prob_v)


def get_prophet_result(df):
    'the same as DIA-NN'

    col_idx = df.columns.str.startswith('score_')
    X = df.loc[:, col_idx].to_numpy()
    y = 1 - df['decoy'].to_numpy()
    print(f'Training the neural network: {(y == 1).sum()} targets, {(y == 0).sum()} decoys, scores num: {sum(col_idx)}')

    # norm
    X = preprocessing.scale(X)

    # train and score
    n_estimator = 12
    X = preprocessing.scale(X)
    mlps = [MLPClassifier(max_iter=1, shuffle=True, random_state=i,
                          learning_rate_init=0.003, solver='adam', batch_size=50,
                          activation='tanh', hidden_layer_sizes=(25, 20, 15, 10, 5)) for i in range(n_estimator)]

    # train and score
    for mlp in mlps:
        mlp.fit(X, y)

    cscore = [mlp.predict_proba(X)[:, 1] for mlp in mlps]
    cscore = np.array(cscore).mean(axis=0)

    # q value
    df['cscore'] = cscore
    df = df.sort_values(by='cscore', ascending=False, ignore_index=True)

    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    decoy_num[decoy_num == 0] = 1
    df['q_value'] = decoy_num / target_num
    df['q_value'] = df['q_value'][::-1].cummin()

    # log
    id_001 = ((df['q_value'] <= 0.001) & (df['decoy'] == 0)).sum()
    id_010 = ((df['q_value'] <= 0.01) & (df['decoy'] == 0)).sum()
    id_100 = ((df['q_value'] <= 0.1) & (df['decoy'] == 0)).sum()
    id_500 = ((df['q_value'] <= 0.5) & (df['decoy'] == 0)).sum()
    print(f'Number of IDs at 50%, 10%, 1%, 0.1% FDR: {id_500}, {id_100}, {id_010}, {id_001}')

    return df


def load_diann_params(dir):
    with open(dir) as f:
        content = f.readlines()
    for line in content:
        line = line.strip().split('\t')

        if line[0].find('MassCorrectionMs1') != -1:
            for i in range(1, len(line)):
                predifine.MassCorrectionMs1.append(float(line[i]))
            continue

        if line[0].find('MassCalCenterMs1') != -1:
            for i in range(1, len(line)):
                predifine.MassCalCenterMs1.append(float(line[i]))
            continue

        if line[0].find('MassCalBinsMs1') != -1:
            predifine.MassCalBinsMs1 = int(line[1])
            continue

        if line[0].find('MassAccuracyMs1') != -1:
            predifine.MassAccuracyMs1 = float(line[1]) * 1000000.
            continue

        if line[0].find('MassCorrection') != -1:
            for i in range(1, len(line)):
                predifine.MassCorrection.append(float(line[i]))
            continue

        if line[0].find('MassCalCenter') != -1:
            for i in range(1, len(line)):
                predifine.MassCalCenter.append(float(line[i]))
            continue

        if line[0].find('MassCalBins') != -1:
            predifine.MassCalBins = int(line[1])
            continue

        if line[0].find('MassAccuracy') != -1:
            predifine.MassAccuracy = float(line[1]) * 1000000.
            continue


@jit(nopython=True)
def predicted_mz_ms1(mz, rt, t, MassCalCenterMs1, MassCalBinsMs1):
    s = t[0] * mz * mz
    if (rt <= MassCalCenterMs1[0]):
        s += t[1] + t[2] * mz
    elif (rt >= MassCalCenterMs1[MassCalBinsMs1 - 1]):
        s += t[1 + (MassCalBinsMs1 - 1) * 2] + t[2 + (MassCalBinsMs1 - 1) * 2] * mz
    else:
        for i in range(1, MassCalBinsMs1):
            if (rt < MassCalCenterMs1[i]):
                u = rt - MassCalCenterMs1[i - 1]
                v = MassCalCenterMs1[i] - rt
                w = u + v
                if w > 0.000000001:
                    s += ((t[1 + (i - 1) * 2] + t[2 + (i - 1) * 2] * mz) * v + (
                            t[1 + i * 2] + t[2 + i * 2] * mz) * u) / w
                break
    return s + mz


@jit(nopython=True)
def predicted_mz(mz, rt, t, MassCalCenter, MassCalBins):
    s = t[0] * mz * mz
    if (rt <= MassCalCenter[0]):
        s += t[1] + t[2] * mz
    elif (rt >= MassCalCenter[MassCalBins - 1]):
        s += t[1 + (MassCalBins - 1) * 2] + t[2 + (MassCalBins - 1) * 2] * mz
    else:
        for i in range(1, MassCalBins):
            if (rt < MassCalCenter[i]):
                u = rt - MassCalCenter[i - 1]
                v = MassCalCenter[i] - rt
                w = u + v
                if w > 0.000000001:
                    s += ((t[1 + (i - 1) * 2] + t[2 + (i - 1) * 2] * mz) * v + (
                            t[1 + i * 2] + t[2 + i * 2] * mz) * u) / w
                break
    return s + mz


def preprocess_info_df(df):
    df['pr_charge'] = df['pr_id'].str[-1].astype(int)

    df['simple_seq'] = df['pr_id'].str[0:-1]
    df['simple_seq'] = df['simple_seq'].replace(['C\(UniMod:4\)', 'M\(UniMod:35\)'], ['C', 'm'], regex=True)
    assert df['simple_seq'].str.contains('UniMod').sum() == 0

    df['seq_len'] = df['simple_seq'].str.len()
    df['second_bone_C'] = df['simple_seq'].str[1].str.upper()
    df['second_bone_N'] = df['simple_seq'].str[-2].str.upper()

    mass_neutron = 1.0033548378
    df['pr_mz_1'] = (df['pr_mz'] * df['pr_charge'] + mass_neutron) / df['pr_charge']
    df['pr_mz_2'] = (df['pr_mz'] * df['pr_charge'] + 2 * mass_neutron) / df['pr_charge']

    # to_numpy
    pr_mz_v = df['pr_mz'].to_numpy()
    pr_mz_1_v = df['pr_mz_1'].to_numpy()
    pr_mz_2_v = df['pr_mz_2'].to_numpy()
    fg_mz_v = df['fg_mz'].to_numpy()
    rt_v = df['rt'].to_numpy()

    pr_mz_pred_v, query_mz_v = [], []

    for i in range(len(df)):
        pr_mz = pr_mz_v[i]
        pr_mz_1 = pr_mz_1_v[i]
        pr_mz_2 = pr_mz_2_v[i]
        query_pr_mz = np.array([pr_mz, pr_mz_1, pr_mz_2])

        # fg_mz from lib
        fg_mz = np.fromstring(fg_mz_v[i], sep=';')

        query_fg_mz = np.concatenate([[pr_mz], fg_mz])

        rt = rt_v[i]  # unit: min
        assert rt < 60. * 4

        query_pr_mz = [predicted_mz_ms1(mz, rt,
                                        np.array(predifine.MassCorrectionMs1),
                                        np.array(predifine.MassCalCenterMs1),
                                        predifine.MassCalBinsMs1) for mz in query_pr_mz]
        query_fg_mz = [predicted_mz(mz, rt,
                                    np.array(predifine.MassCorrection),
                                    np.array(predifine.MassCalCenter),
                                    predifine.MassCalBins) for mz in query_fg_mz]
        pr_mz_pred = query_pr_mz[0]
        query_mz = np.concatenate([query_pr_mz, query_fg_mz])

        pr_mz_pred_v.append(pr_mz_pred)
        query_mz_v.append(query_mz)

    df['pr_mz_pred'] = pr_mz_pred_v
    df['query_mz'] = query_mz_v

    return df


@jit(nopython=True)
def sj_unify_dim(xics_ms1, rts_ms1, xics_ms2, rts_ms2, xics):
    target_dim = xics.shape[1]
    rt_start = rts_ms1[0]
    rt_end = rts_ms1[-1]
    delta_rt = (rt_end - rt_start) / (target_dim - 1)

    idx_ms1 = 1
    idx_ms2 = 1
    for i in range(target_dim):
        x = rt_start + i * delta_rt
        # ms1
        if x > rts_ms1[idx_ms1]:
            idx_ms1 += 1
        x0 = rts_ms1[idx_ms1 - 1]
        x1 = rts_ms1[idx_ms1]
        for j in range(xics_ms1.shape[0]):
            y0 = xics_ms1[j, idx_ms1 - 1]
            y1 = xics_ms1[j, idx_ms1]
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            xics[j, i] = y
        # ms2
        if x > rts_ms2[idx_ms2]:
            idx_ms2 += 1
        x0 = rts_ms2[idx_ms2 - 1]
        x1 = rts_ms2[idx_ms2]
        for j in range(xics_ms2.shape[0]):
            y0 = xics_ms2[j, idx_ms2 - 1]
            y1 = xics_ms2[j, idx_ms2]
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            xics[j + 3, i] = y


def pad_seq_to_mass(simple_seq):
    s = simple_seq.str.cat()
    s = list(s)

    f = operator.itemgetter(*s)
    paded_mass = f(predifine.g_aa_to_mass)

    return paded_mass


@jit(nopython=True)
def find_ok_matches(scan_mz, scan_intensity, mz_query, ppm):
    num_basis = scan_mz.shape[0]
    num_samples = mz_query.shape[0]
    result = np.zeros(len(mz_query), dtype=np.float32)

    for i in range(num_samples):
        sp_i = mz_query[i]
        low = 0
        high = num_basis - 1
        best_j = 0
        if scan_mz[low] == sp_i:
            best_j = low
        elif scan_mz[high] == sp_i:
            best_j = high
        else:
            while high - low > 1:
                mid = int((low + high) / 2)
                if scan_mz[mid] == sp_i:
                    best_j = mid
                    break
                if scan_mz[mid] < sp_i:
                    low = mid
                else:
                    high = mid
            if best_j == 0:
                if abs(scan_mz[low] - sp_i) < abs(scan_mz[high] - sp_i):
                    best_j = low
                else:
                    best_j = high
        # find first match in list !
        while best_j > 0:
            if scan_mz[best_j - 1] == scan_mz[best_j]:
                best_j = best_j - 1
            else:
                break

        mz_nearest = scan_mz[best_j]
        if abs(sp_i - mz_nearest) * 1000000. < ppm * sp_i:
            result[i] = scan_intensity[best_j]

    return result
