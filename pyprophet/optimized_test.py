#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   optimized_test.py
@Author :   Song
@Time   :   2021/4/1 14:43
@Contact:   songjian@westlake.edu.cn
@intro  :   cython函数还是需要好好测试
'''
import numpy as np
import pandas as pd
from pyprophet.optimized import sj_find_ok_matches

if __name__ == '__main__':
    peaks_mz = np.array([500.]).astype(np.float32)
    peaks_int = np.array([23.]).astype(np.float32)
    mz_query = np.array([150., 250., 350.]).astype(np.float32)

    x = sj_find_ok_matches(peaks_mz, peaks_int, mz_query)

    a = 1