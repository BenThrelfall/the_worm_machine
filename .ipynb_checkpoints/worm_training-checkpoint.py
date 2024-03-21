import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from libworm.torch.beta_net import BetaNeuronNet, from_connectome
from libworm.data import connectomes, traces
from libworm import preprocess
from libworm.functions import set_neurons, tcalc_s_inf, set_trace
from libworm.training import basic_train

def main():
    # Set up
    torch.manual_seed(4687)

    trace, _, trace_labels, label2index, timestamps = traces.load_trace()
    timestamps = timestamps - timestamps[0]
    
    chemical, gapjn = connectomes.load_cook_connectome()
    neurons = connectomes.get_main_neurons(chemical, gapjn)
    neurons.sort(key=lambda item: f"AAA{label2index[item]:04d}{item}" if item in label2index else item)
    model = from_connectome(chemical, gapjn, neurons)
    
    cell = "SMBVR"
    
    first_removal = [label2index[key] for key in label2index if key not in neurons]
    trace = np.delete(trace, first_removal, axis=0)
    
    del_index = 0
    size = trace.shape[0]
    
    for i in range(size):
        if i not in label2index.values():
            trace = np.delete(trace, (del_index), axis=0)
        else:
            del_index += 1
    
    voltage = preprocess.trace2volt(trace)
    
    points, labels = preprocess.window_split(voltage, window_size = 16, points_size = 15)
    points = torch.from_numpy(np.squeeze(points))
    labels = torch.from_numpy(np.squeeze(labels))
    
    train_x, test_x, train_y, test_y = train_test_split(points, labels, train_size=0.1)
    
    optimiser = optim.Adam(model.parameters(), lr=0.0001)
    crit = nn.MSELoss()

    # Train
    results = basic_train(model, crit, optimiser,
                    train_x, train_y, neurons,
                    epoches=1, batch=6, timestep=0.005)

if __name__ == "__main__":
    main()

