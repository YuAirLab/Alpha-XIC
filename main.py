#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   main.py
@Author :   Song
@Time   :   2021/1/11 15:16
@Contact:   songjian@westlake.edu.cn
@intro  :   1. read old result 2. extract train data 3. train 4. score
'''
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import ms
import utils
import predifine

import time
from pyprophet import pyprophet_main
import logging

logging.disable(logging.WARN)

def choose_ws(ws):
    ws = Path(ws)
    Path(ws/'osw').mkdir(parents=True, exist_ok=True)
    Path(ws/'osw_alpha').mkdir(parents=True, exist_ok=True)

    lib_path = ws/'lib.tsv'
    ms_path = list(ws.glob('*.mzML'))[0]
    osw_path = ws/'osw'
    out_dir = ws/'osw_alpha'
    return lib_path, ms_path, osw_path, out_dir

if __name__ == '__main__':
    t0 = time.time()

    lib_path, ms_path, osw_path, out_dir = choose_ws(sys.argv[1])

    ## read input
    print(f'loading data and mzml...')
    df_lib = pd.read_csv(lib_path, sep='\t')
    osw_cols = pd.read_csv(osw_path.parent/'osw.tsv', sep='\t', nrows=1).columns.tolist()
    alpha_cols = osw_cols + ['var_alpha_xic_score']
    mzml = ms.load_ms(ms_path, type='DIA')

    # 前处理
    df_lib = utils.get_annos(df_lib)

    ## run pyprophet
    print(f'run PyProphet ...')
    out_path = str(osw_path/'pyprophet')
    cmd = [str(osw_path.parent/'osw.tsv'), '--target.dir=' + out_path]
    df_pypro = pyprophet_main._main(cmd, return_df=True)

    # outliers remove
    bad_peptides_num = (~df_pypro['FullPeptideName'].isin(df_lib['FullUniModPeptideName'])).sum()
    if bad_peptides_num:
        print('number of peptide that not in lib: {}'.format(bad_peptides_num))

    df_pypro = df_pypro[df_pypro['FullPeptideName'].isin(df_lib['FullUniModPeptideName'])]
    df_pypro = df_pypro.reset_index(drop=True)

    # xic_idx
    df_pypro['xic_idx'] = np.arange(len(df_pypro))

    # extract all xics
    t1 = time.time()
    xics_bank, _ = utils.extract_xics(df_pypro, df_lib, mzml)
    del mzml # 后续可能多进程
    print('### extract xics_bank time: {:.2f}s'.format(time.time() - t1))

    id_num_old = utils.get_id_num(df_pypro)
    id_num_now = id_num_old
    for i in range(5):

        print(f'### start training and scoring ... ###')
        t1 = time.time()
        probs_v = []
        for k in range(predifine.ensemble_num):
            xics, y = utils.get_pos_neg_xic(df_pypro, xics_bank, idx_neg=k)
            model = utils.train_model(xics, y, random_state=1234)
            probs = utils.utils_model(df_pypro, xics_bank, model)
            probs_v.append(probs)
        print('### end training and scoring. time: {:.2f}s'.format(time.time() - t1))

        df_pypro['var_alpha_xic_score'] = np.vstack(probs_v).mean(axis=0)

        alpha_path = out_dir/('osw_output_alpha_' + str(i) + '.tsv')
        df_pypro[alpha_cols].to_csv(alpha_path, sep='\t', index=False)

        ## 含alpha-score运行pyprophet，返回df_pypro
        cmd = [str(alpha_path), r'--target.dir=' + str(out_dir/'pyprophet')]
        df_pypro = pyprophet_main._main(cmd, return_df=True)
        df_pypro['xic_idx'] = df_pypro.index.values

        # 鉴定不增长就跳出循环
        id_num_now = utils.get_id_num(df_pypro)
        if id_num_now < id_num_old:
            break
        id_num_old = id_num_now

    logging.disable(logging.DEBUG)
    assert i > 0, 'alpha-xic is invalid'

    alpha_path = out_dir/('osw_output_alpha_' + str(i-1) + '.tsv')
    df = pd.read_csv(alpha_path, sep='\t')
    df.to_csv(out_dir/'osw_output_alpha_final.tsv', sep='\t', index=False)
    cmd = [str(out_dir/'osw_output_alpha_final.tsv'), r'--target.dir=' + str(out_dir/'pyprophet')]
    pyprophet_main._main(cmd, return_df=False)

    print('### All finished. time: {:.2f}min'.format((time.time() - t0)/60.))




