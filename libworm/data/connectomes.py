import pandas as pd

def load_cook_connectome():
    # Import connectome data
    gapjn = {}
    chemical = {}
    
    # Gap Junction
    gap_csv = pd.read_csv("gapjn.csv").fillna(0)
    gap_data = gap_csv.values[:, 1:]
    gap_labels = gap_csv.values[:, 0]
    
    for i, label in enumerate(gap_labels):
        row_dict = {}
        for j, tag in enumerate(gap_labels):
            row_dict[tag] = gap_data[i, j]
    
        gapjn[label] = row_dict
    
    # Chemical 
    chem_csv = pd.read_csv("chemical.csv").fillna(0)
    chem_data = chem_csv.values[:, 1:]
    chem_labels = chem_csv.values[:, 0]
    chem_cols = chem_csv.columns[1:]
    
    for i, label in enumerate(chem_labels):
        row_dict = {}
        for j, tag in enumerate(chem_cols):
            row_dict[tag] = chem_data[i, j]
    
        chemical[label] = row_dict

    return chemical, gapjn

def get_main_neurons(chemical, gapjn, use_gapjn=False):
    queue = {'PLML', 'PLMR'}
    visited = set()
    
    while len(queue) > 0:
    
        neuron = queue.pop()
    
        if neuron not in chemical or neuron not in gapjn:
            continue
        
        visited.add(neuron)
        
        syn = {key for key in chemical[neuron] if chemical[neuron][key] != 0}

        if use_gapjn:
            gap = {key for key in gapjn[neuron] if gapjn[neuron][key] != 0}
        else:
            gap = {}

        comb = syn.union(gap)
        comb = comb.difference(visited)
    
        queue = queue.union(comb)
        
    
    visited = list(visited)
    visited.sort()
    return visited





