from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
import numpy as np
import pandas as pd

def bif_to_csv(input_file):

    # out_file = input_file.split(".")[0]

    reader = BIFReader(input_file)
    model = reader.get_model()

    sampler = BayesianModelSampling(model)
    data = sampler.forward_sample(size=1000000)

    new_df = pd.DataFrame()

    for col in data.columns:
        unique_values = model.get_cpds(col).state_names[col]
        mapping = {unique_values[i]: i for i in range(len(unique_values))}
        new_df[col] = data[col].map(mapping)

        '''
        # If variable is always same value, it cannot provide any info so we do not include
        if data[col].nunique() == 1:
            continue
        # Simple mapping
        if len(unique_values) == 2:
            new_df[col] = (data[col] == unique_values[1]).astype(int)
        
        # One Hot encoding
        if len(unique_values) > 2:
            for val in unique_values[:-1]:
                col_name = f"{col}_{val}"
                new_df[col_name] = (data[col] == val).astype(int)
        '''

    nodes = list(model.nodes())
    new_df = new_df[nodes]
    new_df.to_csv(f"data/hailfinder.csv", index=False)

    # Extract ground truth adjacency matrix from BIF structure
    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Create adjacency matrix (edge from i to j means i -> j)
    true_matrix = np.zeros((num_nodes, num_nodes))
    for parent, child in model.edges():
        i = node_to_idx[parent]
        j = node_to_idx[child]
        true_matrix[i, j] = 1

    # Save true matrix to CSV for comparison
    true_df = pd.DataFrame(true_matrix, index=nodes, columns=nodes)
    true_df.to_csv("output/W_true.csv")

