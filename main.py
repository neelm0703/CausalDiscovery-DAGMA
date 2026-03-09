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

def main():
    utils.set_random_seed(1)
    torch.manual_seed(1)

    hailfinder_path = Path("data/hailfinder.csv")
    wtrue_path = Path("output/W_true.csv")

    # Sample data from bif and get save matrix
    if (not hailfinder_path.exists()) or (not wtrue_path.exists()):
        bif_to_csv("data/hailfinder.bif")


    # Training data
    df = pd.read_csv("data/hailfinder.csv") 
    X = df.values.astype(float)

    # True matrix
    df_W_true = pd.read_csv("output/W_true.csv", index_col=0)
    W_true = df_W_true.to_numpy()


    threshold = 0.3

    # Run linear model (save loss graphs)
    linear_model = DagmaLinear(loss_type='l2')
    linear_W_est = linear_model.fit(X, lambda1=0.04)
    #linear_W_est = linear_model.fit(X, lambda1=0.05, w_threshold=0.1, T=8, mu_init=2, mu_factor=0.9, warm_iter = 3e4, max_iter = 6e4, lr=0.003)
    linear_W_est = (np.abs(linear_W_est) > threshold).astype(float)
    nodes = df.columns
    linear_W_df = pd.DataFrame(linear_W_est, index=nodes, columns=nodes)
    linear_W_df.to_csv("output/DAGMA_linear_W_est.csv")

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

    # Calculate and print final metrics
    linear_metrics = calculate_metrics(linear_W_est, W_true, True)
    print(linear_metrics)
    print(f"Linear SHD: {linear_metrics[0]}")
    print(f"Linear CI_MCC: {linear_metrics[1]}")

    nonlinear_metrics = calculate_metrics(W_est, W_true, True)
    print(nonlinear_metrics)
    print(f"Linear SHD: {nonlinear_metrics[0]}")
    print(f"Linear CI_MCC: {nonlinear_metrics[1]}")

    bn_true = from_numpy_to_bn(W_true)
    bn_linear = from_numpy_to_bn(linear_W_est)

    plot_bayesian_networks(
        [bn_true, bn_linear],
        node_colors=["lightgreen", "lightblue"]
    )

if __name__ == "__main__":
    main()