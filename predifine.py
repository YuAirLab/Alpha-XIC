#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   predifine.py
@Author :   Song
@Time   :   2021/1/11 16:48
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import torch

pos_q = 0.01 # FDR for positive samples
ensemble_num = 1 # number of base learners
extend_time = 0. # s
target_dim = 64

device = torch.device('cuda')
