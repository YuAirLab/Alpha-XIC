#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   train.py
@Author :   Song
@Time   :   2021/1/11 19:26
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils.data
from sklearn.model_selection import train_test_split
from dataloader import Model_Dataset
import predifine
from Model_GRU import Model_GRU
from scipy.signal import savgol_filter

from pyprophet.optimized import sj_adjust_dim

try:
    profile
except:
    profile = lambda x: x


def my_collate(item):
    'list-->pad-->tensor'
    xic_l, fg_num, label = zip(*item)

    max_row = max(fg_num)
    xic = [np.vstack([xics, np.zeros((max_row - len(xics), xics.shape[1]))]) for xics in xic_l]

    xic = torch.tensor(xic)
    fg_num = torch.tensor(fg_num)
    label = torch.tensor(label)

    return xic, fg_num, label


def train_one_epoch(trainloader, model, optimizer, loss_fn):
    model.train()
    epoch_loss_fg = 0.
    device = predifine.device
    for batch_idx, (batch_xic, batch_fg_num, batch_labels) in enumerate(trainloader):
        batch_xic = batch_xic.float().to(device)
        batch_fg_num = batch_fg_num.long().to(device)
        batch_labels = batch_labels.long().to(device)

        # forward
        prob = model(batch_xic, batch_fg_num)

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
    # smooth
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
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               num_workers=2,
                                               shuffle=True, collate_fn=my_collate)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=128,
                                               num_workers=2,
                                               shuffle=True, collate_fn=my_collate)
    # model
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
        print('\r****** test numbers of XICs: {}/{} finished'.format(df_batch.index[-1] + 1, len(df_pypro)), end='',
              flush=True)

        # extract xic
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


def get_annos(df_lib):
    try:
        df = df_lib.sort_values(by=['FullUniModPeptideName', 'PrecursorCharge'], ignore_index=True)
    except:
        df_lib['FullUniModPeptideName'] = df_lib['ModifiedPeptideSequence']
        df = df_lib.sort_values(by=['FullUniModPeptideName', 'PrecursorCharge'], ignore_index=True)

    first_idx = np.where(~df.duplicated(subset=['FullUniModPeptideName', 'PrecursorCharge']))[0]
    first_idx = np.append(first_idx, len(df))
    x = df['ProductMz'].to_numpy().astype(np.float32)
    v = [x[i:j] for i, j in zip(first_idx[:-1], first_idx[1:])]

    df = df.drop_duplicates(subset=['FullUniModPeptideName', 'PrecursorCharge'], ignore_index=True)
    df['ProductMz'] = v
    df['fg_num'] = df['ProductMz'].apply(len)

    return df[['FullUniModPeptideName', 'PrecursorCharge', 'ProductMz', 'fg_num']]


def get_id_num(df):
    return len(df[(df.m_score <= 0.01) &
                  (df.decoy == 0) &
                  (df.peak_group_rank == 1)])


@profile
def extract_xics(df_pypro, df_annos, mzml):
    pd.options.mode.chained_assignment = None

    '''
    get the boundaries of peak groups by df_pypro
    get fragment mz by df_lib
    then extract the xic in mzML
    '''
    df_merge = pd.merge(df_pypro, df_annos,
                        left_on=['FullPeptideName', 'Charge'],
                        right_on=['FullUniModPeptideName', 'PrecursorCharge'])

    xics_v = []

    # broadcast with batch
    for _, df in df_merge.groupby(df_merge.index // 100000):
        print('\rextract xics: {}/{} finished'.format(df.index[-1] + 1, len(df_merge)), end='', flush=True)
        # vectorization
        scans_ms1_rt = mzml.get_ms1_all_rt()
        num_windows = len(mzml.SwathSettings) - 1
        df['query_rt_left'] = df['leftWidth'] - predifine.extend_time
        df['query_rt_right'] = df['rightWidth'] + predifine.extend_time

        df['idx_start'] = np.abs(df['query_rt_left'].to_numpy().reshape(-1, 1) - scans_ms1_rt).argmin(axis=1) * (
                num_windows + 1)
        df['idx_end'] = np.abs(df['query_rt_right'].to_numpy().reshape(-1, 1) - scans_ms1_rt).argmin(axis=1) * (
                num_windows + 1)
        df['ms2_win_idx'] = np.digitize(df['m/z'], mzml.SwathSettings)

        query_mz_v = df['ProductMz'].explode().to_numpy(dtype=np.float32)
        fg_num_v = df['fg_num'].to_numpy()
        fg_idx_v = [0]
        fg_idx_v.extend(fg_num_v)
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

            xics, rts = mzml.get_ms2_xics_by_fg_mz(idx_start, idx_end, ms2_win_idx, query_mz)
            xics = sj_adjust_dim(xics, rts, predifine.target_dim)
            xics_v.append(xics)

    print('\r')
    xics = np.array(xics_v)
    y = (1 - df_merge['decoy']).to_numpy()

    pd.options.mode.chained_assignment = 'warn'

    return xics, y


def get_pos_neg_xic(df_pypro, xics_bank, idx_neg):
    df_pos = df_pypro[(df_pypro.decoy == 0) &
                      (df_pypro.m_score <= predifine.pos_q) &
                      (df_pypro.peak_group_rank == 1)]
    df_neg = df_pypro[(df_pypro.decoy == 1)].sample(n=predifine.ensemble_num * len(df_pos), random_state=1234)
    df_neg = df_neg.iloc[idx_neg * len(df_pos): (idx_neg + 1) * len(df_pos)]

    assert len(df_pos) == len(df_neg)

    df = pd.concat([df_pos, df_neg], axis=0, ignore_index=False)

    # extract
    idx = df['xic_idx'].values
    xics = xics_bank[idx]

    y = (1 - df['decoy']).to_numpy()

    return xics, y
