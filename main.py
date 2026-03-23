import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.bif_to_csv import bif_to_csv
from DAGMA import utils
import torch
from DAGMA.linear import DagmaLinear
from DAGMA.nonlinear import DagmaMLP, DagmaNonlinear
from metrics.metrics import calculate_metrics
from pathlib import Path
from simulation import from_numpy_to_bn, plot_bayesian_networks
from openai import OpenAI

CLIENT = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="aiforge-DZzMR1N2hChWEWTXeVTWr61ms_Em_hcUbznl3l3GaSM",
    timeout=60
)
MODEL_NAMES = [
    "gemma3:4b-it-q8_0"
]



def load_descriptions(path):
    df = pd.read_csv(path)
    return dict(zip(df["variable"], df["description"]))


def get_uncertain_edges(W_raw, nodes, low=0.4, high=0.7):
    edges = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            w = abs(W_raw[i, j])
            if low <= w <= high:
                edges.append((i, j))
    return edges




def query_llm(model, A, B, desc_A, desc_B):
    prompt = f"""
A and B are variables in a causal system. Does A cause B, B cause A, or is there no direct causal relationship? If you are unsure, default to 'No relation'

A represents: {desc_A}
B represents: {desc_B}

Answer ONLY in one of the following formats:
- A->B
- B->A
- No relation
"""

    try:
        response = CLIENT.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in causal reasoning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error with {model}: {e}")
        return None

def main():
    utils.set_random_seed(1)
    torch.manual_seed(1)

    hailfinder_path = Path("data/hailfinder.csv")
    wtrue_path = Path("output/W_true.csv")

    # Sample data from bif and get save matrix
    if (not hailfinder_path.exists()) or (not wtrue_path.exists()):
        bif_to_csv("data/hailfinder.bif")

    desc_map = load_descriptions("hailfinder_descriptions.csv")

    # Training data
    df = pd.read_csv("data/hailfinder.csv") 
    X = df.values.astype(float)
    nodes = df.columns

    # True matrix
    df_W_true = pd.read_csv("output/W_true.csv", index_col=0)
    W_true = df_W_true.to_numpy()


    threshold = 0.4

    # Run linear model (save loss graphs)
    linear_model = DagmaLinear(loss_type='l2')
    linear_W_est = linear_model.fit(X, lambda1=0.04)
    #linear_W_est = linear_model.fit(X, lambda1=0.05, w_threshold=0.1, T=8, mu_init=2, mu_factor=0.9, warm_iter = 3e4, max_iter = 6e4, lr=0.003)
    linear_W_est_raw = linear_W_est.copy()

    # Baseline performance 
    linear_W_est_baseline = (np.abs(linear_W_est) > threshold).astype(float)
    linear_W_est_baseline_df = pd.DataFrame(linear_W_est_baseline, index=nodes, columns=nodes)
    linear_W_est_baseline_df.to_csv("output/DAGMA_linear_W_est.csv")

    linear_metrics = calculate_metrics(linear_W_est_baseline, W_true, True)
    # Dict of LLM assisted SHD and CI_MCC scores - model :(shd, cimcc)
    scores = {"baseline": linear_metrics}

    none_count = 0

    # Use LLM for assitance on uncertain edges
    uncertain_edges = get_uncertain_edges(linear_W_est_raw, nodes)
    votes_on_uncertain_edges = [[0,0,0] for _ in range(len(uncertain_edges))]
    for model in MODEL_NAMES:
      W_est_LLM = linear_W_est_baseline.copy()
      for idx in range(len(uncertain_edges)):
        i,j = uncertain_edges[idx]

        desc_A = desc_map.get(nodes[i], nodes[i])
        desc_B = desc_map.get(nodes[j], nodes[j])
        decision = query_llm(model, "A", "B", desc_A, desc_B)
        if (decision is None):
          decision = "No relation"
          none_count += 1
          
        if "a -> b" in decision.lower() or "a->b" in decision.lower() or "a → b" in decision.lower():
          W_est_LLM[i, j] = 1
          W_est_LLM[j, i] = 0
          votes_on_uncertain_edges[idx][0] += 1
        elif "b -> a" in decision.lower() or "b->a" in decision.lower() or "b → a" in decision.lower():
          W_est_LLM[j, i] = 1
          W_est_LLM[i, j] = 0
          votes_on_uncertain_edges[idx][1] += 1
        else:
          W_est_LLM[i, j] = 0
          W_est_LLM[j, i] = 0
          votes_on_uncertain_edges[idx][2] += 1
      
      linear_metrics_LLM = calculate_metrics(W_est_LLM, W_true, True)
      scores[model] = linear_metrics_LLM

    scores_df = pd.DataFrame.from_dict(scores, orient="index", columns=["SHD", "CI_MCC"])

    print(f"{none_count}/{len(uncertain_edges)} calls to LLM failed")

    print("\nModel Performance\n------------------------------------------------------------------------------------------------\n")
    print(scores_df.to_string())
    scores_df.to_csv("output/scores.csv")

    '''
    plt.figure(figsize=(8,5))
    plt.plot(linear_model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Score (log MSE)")
    plt.title("DAGMA Linear Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/linear_training_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(linear_model.cimcc_history)
    plt.xlabel("Checkpoint")
    plt.ylabel("CI_MCC")
    plt.title("DAGMA Linear Training CI_MCC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/linear_training_cimcc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(linear_model.shd_history)
    plt.xlabel("Checkpoint")
    plt.ylabel("SHD")
    plt.title("DAGMA Linear Training SHD Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/linear_training_shd.png", dpi=300)
    plt.close()
    '''

    '''
    # Run non linear model (save loss graphs)
    n, d = X.shape 
    eq_model = DagmaMLP(dims=[d, 10, 1], bias=True)
    nonlinear_model = DagmaNonlinear(eq_model)
    W_est = nonlinear_model.fit(X, lambda1=0.02, lambda2=0.005)
    W_est = (np.abs(W_est) > threshold).astype(float)
    nodes = df.columns
    W_df = pd.DataFrame(W_est, index=nodes, columns=nodes)
    W_df.to_csv("output/DAGMA_nonlinear_W_est.csv")

    plt.figure(figsize=(8,5))
    plt.plot(nonlinear_model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Score (log MSE)")
    plt.title("DAGMA Non-Linear Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/DAGMA_nonlinear_training_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(nonlinear_model.cimcc_history)
    plt.xlabel("Checkpoint")
    plt.ylabel("CI_MCC")
    plt.title("DAGMA Non-Linear Training CI_MCC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/DAGMA_nonlinear_training_cimcc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(nonlinear_model.shd_history)
    plt.xlabel("Checkpoint")
    plt.ylabel("SHD")
    plt.title("DAGMA Non-Linear Training SHD Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/DAGMA_nonlinear_training_shd.png", dpi=300)
    plt.close()
    '''

    '''
    # Calculate and print final metrics
    linear_metrics = calculate_metrics(linear_W_est, W_true, True)
    print(linear_metrics)
    print(f"Linear SHD: {linear_metrics[0]}")
    print(f"Linear CI_MCC: {linear_metrics[1]}")
    '''

    '''
    nonlinear_metrics = calculate_metrics(W_est, W_true, True)
    print(nonlinear_metrics)
    print(f"Linear SHD: {nonlinear_metrics[0]}")
    print(f"Linear CI_MCC: {nonlinear_metrics[1]}")
    '''

    '''
    bn_true = from_numpy_to_bn(W_true)
    bn_linear = from_numpy_to_bn(linear_W_est)

    plot_bayesian_networks(
        [bn_true, bn_linear],
        node_colors=["lightgreen", "lightblue"]
    )
    '''

if __name__ == "__main__":
    main()