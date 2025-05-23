import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from libworm.data.neuron import gaba_list, full_sensory_list

###
# Gap Junctions and Synapses
###

# Synapse coefficent
def co_syn(big_G_syn, big_s):
    sum = 0
    for j in range(len(big_s)):
        sum += big_G_syn[:, j] * big_s[j]
    return sum    

# Synapse intercept
def int_syn(big_G_syn, big_s, big_E):
    
    sum = 0
    for j in range(len(big_s)):
        sum += big_G_syn[:, j] * big_s[j] * big_E[:, j]
    return sum    

# Synapse current
def I_syn(V_m, co, int):
    return V_m * co - int

# Gapjn coefficent
def co_gap(big_G_gap):
    return np.sum(big_G_gap, axis=1)

# Gapjn intercept
def int_gap(big_G_gap, big_V):
    sum = 0
    for j in range(len(big_V)):
        sum += big_G_gap[:, j] * big_V[j]

    return sum

# Gapjn current
def I_gap(V_m, co, int):
    return V_m * co - int


# Voltage limit
def V_inf(co_syn, int_syn, co_gap, int_gap, I_in):
    GE_leak = -350
    G_leak = 10

    top = GE_leak + int_syn + int_gap + I_in
    bottom = G_leak + co_syn + co_gap

    return top / bottom

def delta_s(V_m, s_i):
    a_r = 1.0
    a_d = 5.0
    beta = 0.125
    V_th = -15.0 #??

    sig = 1.0 / (1.0 + np.exp(-beta * (V_m - V_th)))

    return a_r * sig * (1.0 - s_i) - a_d * s_i

# Simple/direct version
def new_s(V_m):
    a_r = 1.0
    a_d = 5.0
    beta = 0.125
    V_th = -15.0 #??
    sig = 1.0 / (1.0 + np.exp(-beta * (V_m - V_th)))

    return (a_r * sig) / (a_r * sig + a_d)

###
# Leak Current and Voltage
###

def I_leak(V_m, G_leak = 10.0 ,E_leak = -35.0):
    return G_leak * (V_m - E_leak)

def delta_V_m(V_m, I_leak, I_gap, I_syn, I_in):
    C_m = 1.0 #pF
    current_sum = -I_leak - I_gap - I_syn + I_in
    return current_sum / C_m

###
# Joint and Simulation
###

class NeuronNetwork:
    def __init__(self, big_V, big_G_syn, big_G_gap, big_E = None, big_s = None, v_clamp=None, labels=[], G_leak = 10.0, E_leak = -35.0):

        self.labels = labels

        self.store_data = True

        if len(labels) > 0:
            self.neurons = {}

            for i in range(len(labels)):
                self.neurons[labels[i]] = i 
        
        self.G_leak = G_leak
        self.E_leak = E_leak
        
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
            self.big_E = np.array([[0] * len(big_V) for V_m in big_V])
        else:
            self.big_E = big_E

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
        self.V_inf_store = []

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

    def step(self, time_step, input_current, limiter=True, direct_s=False):

        # Calculate deltas

        leak_current = I_leak(self.big_V, self.G_leak, self.E_leak)

        synapse_coeffiecnt = co_syn(self.big_G_syn, self.big_s)
        synapse_intercept = int_syn(self.big_G_syn, self.big_s, self.big_E)
        synapse_current = I_syn(self.big_V, synapse_coeffiecnt, synapse_intercept)

        gap_coefficent = co_gap(self.big_G_gap)
        gap_intercept = int_gap(self.big_G_gap, self.big_V)
        gap_current = I_gap(self.big_V, gap_coefficent, gap_intercept)

        V_infinity = V_inf(synapse_coeffiecnt, synapse_intercept, gap_coefficent, gap_intercept, input_current)
        d_V_m = delta_V_m(self.big_V, leak_current, gap_current, synapse_current, input_current)
        d_s = delta_s(self.big_V, self.big_s)

        voltage_step = (self.v_clamp * d_V_m) * time_step
        
        if limiter:
            # Clamp
            V_diff = np.abs(V_infinity - self.big_V)
            voltage_step = np.clip(voltage_step, -V_diff, V_diff)
            

        # Update
        self.time += time_step

        if direct_s:
            self.big_s = new_s(self.big_V)
        else:
            self.big_s += d_s * time_step
            
        self.big_V += voltage_step
        
        
        #Store
        if(self.store_data):
            self.t_store.append(self.time)
            self.V_store.append(self.big_V.copy())
            self.s_store.append(self.big_s.copy())
            self.leak_store.append(leak_current.copy())
            self.in_store.append(input_current.copy())
            self.syn_store.append(synapse_current.copy())
            self.gap_store.append(gap_current.copy())
            self.V_inf_store.append(V_infinity.copy())
    
    def adv_run(self, delta_t, run_time, current_gen, show_progress=True, limiter=True):

        time_range = int(run_time / delta_t)

        for i in range(time_range):
            if show_progress and i % (time_range // 10) == 0:
                print("#", end="")
                
            input_current = current_gen(self.time)
            self.step(delta_t, input_current, limiter=limiter)

    def simple_run(self, delta_t, run_time, show_progress=True, limiter=True, direct_s=False):
        time_range = int(run_time / delta_t)
        input_current = np.array([0 for i in range(len(self.big_V))])

        for i in range(time_range):
            if show_progress and i % (time_range // 10) == 0:
                print("#", end="")
                
            self.step(delta_t, input_current, limiter, direct_s)

    def report(self):
        V_max = np.argmax(self.big_V)
        V_min = np.argmax(-self.big_V)

        print(f"Neurons {len(self.big_V)} ({len(self.labels)})")
        print(f"V_max = {self.big_V[V_max]} ({V_max})")
        print(f"V_min = {self.big_V[V_min]} ({V_min})")

    def show_all_data(self, start=0, end=None):

        if end is None:
            end = len(self.t_store)

        voltage = np.array(self.V_store)
        leak = np.array(self.leak_store)
        input = np.array(self.in_store)
        synapse = np.array(self.syn_store)

        gates = np.array(self.s_store)

        # Voltage time curves
        for i in range(len(self.big_V)):
            plt.plot(self.t_store[start:end], voltage[start:end, i], label=f'V_m_{i}')
        plt.legend(loc='best')
        plt.show()

        # Synapse gates
        for i in range(len(self.big_V)):
            plt.plot(self.t_store[start:end], gates[start:end, i], label=f's_{i}')
        plt.legend(loc='best')
        plt.show()
        
        # Current curves
        for i in range(len(self.big_V)):
            plt.plot(self.t_store[start:end-1], leak[start:end, i], label=f'I_leak_{i}')
            plt.plot(self.t_store[start:end-1], input[start:end, i], label=f'I_in_{i}')
            plt.plot(self.t_store[start:end-1], synapse[start:end, i], label=f'I_syn_{i}')
            plt.legend(loc='best')
            plt.show()

    def show_large_voltage_data(self, count = None, per = 8.0, stride=1):

        if count is None:
            voltage = np.array(self.V_store)
        else:
            voltage = np.array(self.V_store)[:, :count]

        size = int(np.ceil(np.sqrt(voltage.shape[1] / per)))

        fig, axs = plt.subplots(ncols=size, nrows=size, figsize=(5*12, 5*12))

        ax_index = -1
        
        # Voltage time curves
        for i in range(voltage.shape[1]):
            if i % per == 0:
                ax_index += 1
                
            axs[ax_index // size, ax_index % size].plot(self.t_store[::stride], voltage[::stride, i], label=f'V_m_{i}')
            
        plt.show()

    def show_data(self, n, start=0, end=None):

        if type(n) is str:
            n  = self.neurons[n]

        if end is None:
            end = len(self.t_store)

        voltage = np.array(self.V_store)
        V_inf = np.array(self.V_inf_store)
        leak = np.array(self.leak_store)
        input = np.array(self.in_store)
        synapse = np.array(self.syn_store)
        syn_gate = np.array(self.s_store)
        gap = np.array(self.gap_store)

        plt.plot(self.t_store[start:end], voltage[start:end, n], label=f'V_m_{n}')
        plt.plot(self.t_store[start:end-1], V_inf[start:end, n], label=f'V_inf_{n}')
        plt.legend(loc='best')
        plt.show()

        plt.plot(self.t_store[start:end], syn_gate[start:end, n], label=f's_{n}')
        plt.show()
        
        plt.plot(self.t_store[start:end-1], leak[start:end, n], label=f'I_leak_{n}')
        plt.plot(self.t_store[start:end-1], input[start:end, n], label=f'I_in_{n}')
        plt.plot(self.t_store[start:end-1], synapse[start:end, n], label=f'I_syn_{n}')
        plt.plot(self.t_store[start:end-1], gap[start:end, n], label=f'I_gap_{n}')
        plt.legend(loc='best')
        plt.show()

def from_connectome(chemical, gapjn, neurons,
                    G_syn_value = 100.0, 
                    E_syn_ex_value = 0.0,
                    E_syn_in_value = -45.0,
                    G_gap_value = 100.0,
                    G_leak_value = 10.0,
                    E_leak_value = -35.0,
                    V_value = -35.0):
    
    G_syn = []
    E_syn = []
    G_gapjn = []

    # TODO: Check this is the correct order
    for cell in neurons:
        syn_E = E_syn_ex_value if cell not in gaba_list else E_syn_in_value
        E_syn.append([syn_E] * len(neurons))
    
    for i, to_cell in enumerate(neurons):
        G_syn.append([])
        G_gapjn.append([])
        for j, from_cell in enumerate(neurons):
            syn_value = G_syn_value if chemical[from_cell][to_cell] > 0 else 0.0
            G_syn[i].append(syn_value)
    
            gap_value = G_gap_value if gapjn[from_cell][to_cell] > 0 else 0.0
            G_gapjn[i].append(gap_value)
    
    G_syn = np.array(G_syn)
    E_syn = np.array(E_syn)
    G_gapjn = np.array(G_gapjn)
    G_leak = np.array([G_leak_value for cell in neurons])
    E_leak = np.array([E_leak_value for cell in neurons])
    big_V = np.array([V_value for cell in neurons])
    

    return NeuronNetwork(big_V, G_syn, G_gapjn, E_syn, labels=neurons, G_leak = G_leak, E_leak = E_leak)





