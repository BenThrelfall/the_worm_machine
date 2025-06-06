import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

###
# Gap Junctions and Synapses
###

def I_gap(i, big_V, big_G_gap):
    sum = 0
    for j in range(len(big_V)):
        sum += big_G_gap[i, j] * (big_V[i] - big_V[j])

    return sum

def I_syn(i, big_V, big_E, big_s, big_G_syn):
    sum = 0
    for j in range(len(big_V)):
        sum += big_G_syn[i, j] * big_s[j] * (big_V[i] - big_E[j])
    return sum

def delta_s(V_m, s_i):
    a_r = 1
    a_d = 5
    beta = 0.125
    V_th = -15 #??

    sig = 1 / (1 + np.exp(-beta * (V_m - V_th)))

    return a_r * sig * (1 - s_i) - a_d * s_i

# Simple/direct version
def new_s(V_m):
    a_r = 1
    a_d = 5
    beta = 0.125
    V_th = -15 #??
    sig = 1 / (1 + np.exp(-beta * (V_m - V_th)))

    return (a_r * sig) / (a_r * sig + a_d)

###
# Leak Current and Voltage
###

def I_leak(V_m):
    E_leak = -35 #mV
    G_leak = 10 #pS I think

    return G_leak * (V_m - E_leak)

def delta_V_m(V_m, I_leak, I_gap, I_syn, I_in):
    C_m = 1 #pF

    current_sum = -I_leak - I_gap - I_syn + I_in
    return current_sum / C_m

###
# Joint and Simulation
###

class NeuronNetwork:
    def __init__(self, big_V, big_G_syn, big_G_gap, big_E = None, big_s = None, v_clamp=None, labels=None):

        self.labels = labels

        if len(labels) > 0:
            self.neurons = {}

            for i in range(len(labels)):
                self.neurons[labels[i]] = i 
        
        # Flags
        self.disable_gapjn = False
        self.disable_syn = False
        self.disable_leak = False
        
        # Time
        self.time = 0.0
        
        # Voltage
        self.big_V = big_V

        # Synapse gates
        if big_s is None:
            self.big_s = np.array([new_s(V_m) for V_m in big_V])
        else:
            self.big_s = big_s

        # Synapse conductances
        self.big_G_syn = big_G_syn

        # Gap conductances
        self.big_G_gap = big_G_gap

        if big_E is None:
            self.big_E = np.array([0 for V_m in big_V])
        else:
            self.big_E = big_E

        # Indices (kinda silly)
        self.indices = [i for i in range(len(big_V))]

        # Voltage Clamping
        if v_clamp is None:
            self.v_clamp = np.array([1 for i in big_V])
        else:
            self.v_clamp = v_clamp
        
        # Storage

        self.t_store = [self.time] # Time
        self.V_store = [self.big_V.copy()] # Voltage
        self.s_store = [self.big_s.copy()] # synapse gates
        self.leak_store = [] # leak current
        self.syn_store = [] # synapse current
        self.gap_store = [] # gap curret 
        self.in_store = [] # input current

    
    def get_neuron(self, name):
        index = self.neurons[name]
        return (self.big_V[index], self.big_s[index])

    def set_neuron(self, name, v=None, s=None):

        index = self.neurons[name]
        
        if not s:
            s = self.big_s[index]
        if not v:
            v = self.big_V[index]

        self.big_V[index] = v
        self.big_s[index] = s

    
    def step(self, time_step, input_current):

        # Calculate deltas
        leak_current = I_leak(self.big_V) if not self.disable_leak else np.zeros_like(self.big_V)
        gap_current = I_gap(self.indices, self.big_V, self.big_G_gap) if not self.disable_gapjn else np.zeros_like(self.big_V)
        syn_current = I_syn(self.indices, self.big_V, self.big_E, self.big_s, self.big_G_syn) if not self.disable_syn else np.zeros_like(self.big_V)
        
        d_V_m = delta_V_m(self.big_V, leak_current, gap_current, syn_current, input_current)
        d_s = delta_s(self.big_V, self.big_s)
        

        # Update
        self.time += time_step
        self.big_s += d_s * time_step
        self.big_V += (self.v_clamp * d_V_m) * time_step
        
        
        #Store
        self.t_store.append(self.time)
        self.V_store.append(self.big_V.copy())
        self.s_store.append(self.big_s.copy())
        self.leak_store.append(leak_current.copy())
        self.in_store.append(input_current.copy())
        self.syn_store.append(syn_current.copy())
        self.gap_store.append(gap_current.copy())
    
    def adv_run(self, delta_t, run_time, current_gen, show_progress=True):

        time_range = int(run_time / delta_t)

        for i in range(time_range):
            if show_progress and i % (time_range // 10) == 0:
                print("#", end="")
                
            input_current = current_gen(self.time)
            self.step(delta_t, input_current)

    def simple_run(self, delta_t, run_time, show_progress=True):
        time_range = int(run_time / delta_t)
        input_current = np.array([0 for i in range(len(self.big_V))])

        for i in range(time_range):
            if show_progress and i % (time_range // 10) == 0:
                print("#", end="")
                
            self.step(delta_t, input_current)

    def report(self):
        V_max = np.argmax(self.big_V)

        print(f"V_max = {self.big_V[V_max]} ({V_max})")

    def show_all_data(self, start=0, end=None):

        if end is None:
            end = len(self.t_store)

        voltage = np.array(self.V_store)
        leak = np.array(self.leak_store)
        input = np.array(self.in_store)
        synapse = np.array(self.syn_store)

        # Voltage time curves
        for i in range(len(self.big_V)):
            plt.plot(self.t_store[start:end], voltage[start:end, i], label=f'V_m_{i}')
        plt.legend(loc='best')
        plt.show()
        
        for i in range(len(self.big_V)):
            plt.plot(self.t_store[start:end-1], leak[start:end, i], label=f'I_leak_{i}')
            plt.plot(self.t_store[start:end-1], input[start:end, i], label=f'I_in_{i}')
            plt.plot(self.t_store[start:end-1], synapse[start:end, i], label=f'I_syn_{i}')
            plt.legend(loc='best')
            plt.show()

    def show_data(self, n, start=0, end=None):

        if end is None:
            end = len(self.t_store)

        voltage = np.array(self.V_store)
        leak = np.array(self.leak_store)
        input = np.array(self.in_store)
        synapse = np.array(self.syn_store)
        gap = np.array(self.gap_store)

        plt.plot(self.t_store[start:end], voltage[start:end, n], label=f'V_m_{n}')
        plt.legend(loc='best')
        plt.show()
        
        plt.plot(self.t_store[start:end-1], leak[start:end, n], label=f'I_leak_{n}')
        plt.plot(self.t_store[start:end-1], input[start:end, n], label=f'I_in_{n}')
        plt.plot(self.t_store[start:end-1], synapse[start:end, n], label=f'I_syn_{n}')
        plt.plot(self.t_store[start:end-1], gap[start:end, n], label=f'I_gap_{n}')
        plt.legend(loc='best')
        plt.show()



