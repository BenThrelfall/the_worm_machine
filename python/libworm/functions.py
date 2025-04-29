import numpy as np
import torch

def calc_s_inf(V_m):
    a_r = 1
    a_d = 5
    beta = 0.125
    V_th = -15 #??
    sig = 1 / (1 + np.exp(-beta * (V_m - V_th)))

    return (a_r * sig) / (a_r * sig + a_d)

def tcalc_s_inf(V_m):
    a_r = 1
    a_d = 5
    beta = 0.125
    V_th = -15 #??
    sig = 1 / (1 + torch.exp(-beta * (V_m - V_th)))

    return (a_r * sig) / (a_r * sig + a_d)

def set_neurons(voltage, trace, labels, label2index):
    for i, cell in enumerate(labels):
        if cell not in label2index:
            continue
        index = label2index[cell]
        voltage[:, i] = trace[:, index]

def set_trace(voltage, trace, layer, labels, label2index):
    for i, cell in enumerate(labels):
        if cell not in label2index:
            continue
        index = label2index[cell]
        trace[:, index, layer] = voltage[:, i]