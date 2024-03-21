import numpy as np
import torch
from torch import nn, optim
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from libworm.functions import tcalc_s_inf

def basic_train(model, criterion, optimizer, points, labels,
          cell_labels, epoches=5, batch=64,
          timestep=0.001, data_timestep=0.60156673, do_print=True):

    dataset = TensorDataset(points, labels)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    
    total_start_time = time.time()
    loss = -1
    for i in range(1,epoches+1):
        for points_batch, labels_batch in dataloader:
            start_time = time.time()
            optimizer.zero_grad()
            
            sim_time = 0.0
            next_timestamp = 0.0

            voltage = torch.ones((points_batch.shape[0], len(cell_labels))) * -20.0
            gates = torch.ones((points_batch.shape[0], len(cell_labels))) * -20.0

            
            #Prepare
            for i in range(points_batch.shape[2]):
                while True:
                    if sim_time >= next_timestamp:
                        inter = F.pad(points_batch[:, :, i], (0, voltage.shape[1] - points_batch.shape[1]), "constant", 0.0)
                        voltage = voltage.where(inter == 0.0, inter)
                        gates = tcalc_s_inf(voltage)
                        next_timestamp += data_timestep
                        break;
                    voltage, gates,_,_ = model(voltage, gates, timestep)
                    sim_time += timestep

            while sim_time < next_timestamp:
                voltage, gates,_,_ = model(voltage, gates, timestep)
                sim_time += timestep


            final_output = voltage[:, :labels_batch.shape[1]]
            
            #Compare
            loss = criterion(final_output, labels_batch)
            loss.backward()
            
            optimizer.step()
            
            end_time = time.time()
            print(f"Batch Complete: Time {start_time - end_time} Loss: {loss.item()}")
        
        

            
    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    if(do_print):
        print(f"Total Time {total_time_taken}")
        
    final_loss = loss.item()
    
    return (total_time_taken, final_loss)