import numpy as np
import torch
from torch import nn, optim

class BetaNeuronNet(nn.Module):
    def __init__(self, G_leak, E_leak, G_syn, E_syn, G_gap):
        super().__init__()

        self.G_leak = nn.Parameter(torch.from_numpy(G_leak))
        self.E_leak = nn.Parameter(torch.from_numpy(E_leak))

        self.G_syn = nn.Parameter(torch.from_numpy(G_syn))
        self.E_syn = nn.Parameter(torch.from_numpy(E_syn))

        self.G_gap = nn.Parameter(torch.from_numpy(G_gap))

    def calc_co_syn(self, big_s):
        sum = 0
        for j in range(len(big_s)):
            sum += self.G_syn[:, j] * big_s[j]
        return sum    

    def calc_int_syn(self, big_s):
        sum = 0
        for j in range(len(big_s)):
            sum += self.G_syn[:, j] * big_s[j] * self.E_syn[j]
        return sum    

    def calc_I_syn(self, Voltage, co, int):
        return Voltage * co - int

    # Gapjn coefficent
    def calc_co_gap(self):
        return torch.sum(self.G_gap, dim=1)
    
    # Gapjn intercept
    def calc_int_gap(self, voltage):
        sum = 0
        for j in range(len(voltage)):
            sum += self.G_gap[:, j] * voltage[j]
    
        return sum
    
    # Gapjn current
    def calc_I_gap(self, voltage, co, int):
        return voltage * co - int

    def calc_I_leak(self, Voltage):    
        return self.G_leak * (Voltage - self.E_leak)

    def calc_delta_V(self, voltage, I_leak, I_gap, I_syn):
        current_sum = -I_leak - I_syn - I_gap
        return current_sum

    def calc_delta_s(self, Voltage, gate):
        a_r = 1
        a_d = 5
        beta = 0.125
        V_th = -15 #??
    
        sig = 1 / (1 + torch.exp(-beta * (Voltage - V_th)))
    
        return a_r * sig * (1 - gate) - a_d * gate

    def calc_V_inf(self, co_syn, int_syn, co_gap, int_gap):
        GE_leak = self.G_leak * self.E_leak
        G_leak = self.G_leak
    
        top = GE_leak + int_syn + int_gap
        bottom = G_leak + co_syn + co_gap
    
        return top / bottom
        
    def forward(self, big_V, big_s, time_step):

        leak_current = self.calc_I_leak(big_V)

        syn_co = self.calc_co_syn(big_s)
        syn_int = self.calc_int_syn(big_s)
        syn_current = self.calc_I_syn(big_V, syn_co, syn_int)

        gap_co = self.calc_co_gap()
        gap_int = self.calc_int_gap(big_V)
        gap_current = self.calc_I_gap(big_V, gap_co, gap_int)

        delta_V = self.calc_delta_V(big_V, leak_current, gap_current, syn_current)
        delta_s = self.calc_delta_s(big_V, big_s)

        V_inf = self.calc_V_inf(syn_co, syn_int, gap_co, gap_int)

        voltage_step = delta_V * time_step
        
        V_diff = torch.abs(V_inf - big_V)
        voltage_step = torch.clamp(voltage_step, -V_diff, V_diff)

        new_V = big_V + voltage_step
        new_s = big_s + (delta_s * time_step)
        
        return new_V, new_s, leak_current, syn_current 