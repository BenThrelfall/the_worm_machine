import numpy as np

def window_split(trace, window_size = 30, points_size = 15):
    trace_window = []
    
    for i in range(0, trace.shape[1]-window_size):
        trace_window.append(trace[:, i:i+window_size])
    
    trace_window = np.array(trace_window)

    points = trace_window[:, :, :points_size]
    labels = trace_window[:, :, points_size:]

    return points, labels

def trace2volt(trace, base = -30.0, co = 5.0):
    return base + co * trace