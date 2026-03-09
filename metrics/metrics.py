import numpy as np
import networkx as nx
import pandas as pd

# from true_cpdag import convert_cpdag_to_01
from metrics.utils import true_d_sep



def shd(dag_pred: np.ndarray, dag_true: np.ndarray) -> int:
    """
    Calculate the Structural Hamming Distance (SHD) between the predicted DAG and the ground truth DAG.
    The SHD is the number of edges that need to be added or removed to transform one graph into the other.
    """
    assert dag_pred.shape == dag_true.shape, "DAG shapes do not match"
    assert dag_pred.ndim == 2, "DAG must be a 2D array"
    assert dag_pred.shape[0] == dag_pred.shape[1], "DAG must be a square matrix"

    # Compute the number of differences
    shd = np.sum(np.abs(dag_pred - dag_true))

    return int(shd)



def ci_mcc(pred_dag, true_ci=None, true_dag=None):
    """
    Compute the Matthews Correlation Coefficient (MCC) between the 0th- and 1st-order d-separation statements implied by
        the predicted DAG and those given in true_ci.

    Args:
        pred_dag (np.ndarray): The predicted DAG, of shape (n, n).
        true_ci (np.ndarray): Tthe ground-truth 0th- and 1st-order d-separation statements, of shape (n+1, n, n).
                              The first slice is the 0th-order ci, and the rest are the 1st-order ci.
        true_dag (np.ndarray): Alternatively, the ground-truth DAG, of shape (n, n).
    """
    n_nodes = pred_dag.shape[0]
    # Convert to networkx graphs
    pred_G = nx.from_numpy_array(pred_dag, create_using=nx.DiGraph)
    # Get the d-separation statements from these 2 graphs
    pred_dsep = true_d_sep(pred_G, n_nodes).numpy()   # (n+1, n, n)
    if true_ci is None:
        true_G = nx.from_numpy_array(true_dag, create_using=nx.DiGraph)
        # Get the d-separation statements from the ground truth graph
        true_ci = true_d_sep(true_G, n_nodes).numpy()
    # Flatten the statements
    pred = pred_dsep.flatten()
    true = true_ci.flatten()

    # Calculate weighted confusion matrix components
    mask_1 = (pred == 1)
    mask_0 = (pred == 0)
    
    tp = np.sum(true[mask_1])          # True positives (confidence in class 1)
    fp = np.sum(1 - true[mask_1])      # False positives (penalty for class 0)
    tn = np.sum(1 - true[mask_0])      # True negatives (confidence in class 0)
    fn = np.sum(true[mask_0])          # False negatives (penalty for class 1)
    
    # Calculate MCC components
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    # Handle edge case with zero denominator
    if denominator == 0:
        return 0.0
    return (numerator / denominator).item()

'''
def calculate_metrics():
    output =[]

    df_W_est = pd.read_csv("output/DAGMA_linear_W_est.csv", index_col=0)
    df_W_true = pd.read_csv("output/W_true.csv", index_col=0)

    W_est = df_W_est.to_numpy()
    W_true = df_W_true.to_numpy()

    shd_score = shd(W_est, W_true)
    ci_mcc_score = ci_mcc(W_est, None, W_true)
    output.append(shd_score)
    output.append(ci_mcc_score)

    df_W_est = pd.read_csv("output/DAGMA_nonlinear_W_est.csv", index_col=0)
    W_est = df_W_est.to_numpy()

    shd_score = shd(W_est, W_true)
    ci_mcc_score = ci_mcc(W_est, None, W_true)
    output.append(shd_score)
    output.append(ci_mcc_score)

    return output
'''


def calculate_metrics(W_est, W_true, flag):
    output = []
    shd_score = shd(W_est, W_true)
    output.append(shd_score)

    if (flag):
        ci_mcc_score = ci_mcc(W_est, None, W_true)
        output.append(ci_mcc_score)
    else:
        output.append(1)

    return output