import json
import numpy as np

def load_trace(file_name="data/2022-08-02-01.json"):
    # Import neuron data
    with open(file_name, "r") as file:
        data_text = file.read()
        data_dict = json.loads(data_text)
        neuron_labels = data_dict["labeled"]
        neuron_trace = np.array(data_dict["trace_original"], dtype=np.single).T
        neuron_z_trace = np.array(data_dict["trace_array"], dtype=np.single)
        timestamps = np.array(data_dict["timestamp_confocal"], dtype=np.single)
    
    with open("neuron_list.txt", "r") as file:
        trace_labels = [line.strip() for line in file.readlines()]
    
    label2index = {}
    
    for key in neuron_labels:
        label = neuron_labels[key]['label']
        if '?' in label:
            continue
        label2index[label] = int(key) - 1

    return neuron_trace, neuron_z_trace, neuron_labels, label2index, timestamps