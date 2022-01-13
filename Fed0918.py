#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def traditionalFedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
    
    
def FedAvg(w, z_cluster):
    w_avg = copy.deepcopy(w[0])
    total_users = 0
    for k in z_cluster.keys():
        total_users += len(z_cluster[k])    
    z_cluster_att = {}
    for k in z_cluster.keys():
        att = len(z_cluster[k]) / total_users 
        z_cluster_att[k] = att        
    for k in w_avg.keys():
        for i in range(len(w)):
            w_avg[k] += w[i][k]*z_cluster_att[i]
    return w_avg
