#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   main.py
@Author :   Song
@Time   :   2021/1/11 15:16
@Contact:   songjian@westlake.edu.cn
'''
import numpy as np
import pandas as pd
from pathlib import Path

import ms
import utils
import time
import sys

def choose_ws(ws):
    ws = Path(r"G:\000_Alpha_XIC\experiments") / ws
    ms_path = list(ws.glob('*.mzML'))[0]
    path_params = ws / 'cal_params.tsv'
    path_diann = ws / 'diann_out.tsv'
    path_info = ws / 'diann_info.tsv'
    path_info_q = ws / 'alpha_out.tsv'
    return path_params, ms_path, path_diann, path_info, path_info_q


if __name__ == '__main__':
    t0 = time.time()

    path_params, ms_path, path_diann, path_info, path_info_q = choose_ws(sys.argv[1])
    utils.load_diann_params(path_params)

    # read info input
    t = time.time()
    print(f'loading diann_info.tsv and calculate calibrated mz...')
    df = pd.read_csv(path_info, sep='\t')
    raw_cols = list(df.columns)
    df = utils.preprocess_info_df(df)

    # load mz
    mzml = ms.load_ms(ms_path, type='DIA')

    # add xic idx
    df['xic_idx'] = np.arange(len(df))

    # extract xic
    xics_bank, _ = utils.extract_diann_xics(df, mzml)
    del mzml

    ## train
    print(f'Alpha-XIC is working ...')
    xics, y = utils.get_diann_pos_neg_xic(df, xics_bank)
    model = utils.train_model(xics, y, random_state=1234)
    probs = utils.utils_model(df, xics_bank, model)

    df['score_alpha_xic'] = probs

    # statistical validation
    df_result = utils.get_prophet_result(df)

    cols = raw_cols + ['score_alpha_xic', 'cscore', 'q_value', 'Precursor.Quantity']

    print('### All finished. time: {:.2f}min'.format((time.time() - t0) / 60.))

    # save
    df_diann = pd.read_csv(path_diann, sep='\t', usecols=['Precursor.Id', 'Precursor.Quantity'])
    df_result = pd.merge(df_result, df_diann, left_on='pr_id', right_on='Precursor.Id')
    df_result[cols].to_csv(path_info_q, sep='\t', index=False)
