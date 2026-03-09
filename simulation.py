import os, sys, random, itertools, pickle, argparse, multiprocessing, gc
import pyagrum as gum
import numpy as np
from DAGMA.utils import simulate_dag, simulate_parameter, simulate_linear_sem, simulate_nonlinear_sem
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


def plot_bayesian_networks(bns, 
        with_labels=True,
        node_size=500,
        node_colors=["lightblue", "lightgreen"],
        font_size=10,
        font_weight="bold",
        arrowsize=10,):
    """
    Plots two Bayesian Networks side by side while maintaining consistent node coordinates.
    
    Args:
        bn1 (gum.BayesNet): The first Bayesian network.
        bn2 (gum.BayesNet): The second Bayesian network.
    """
    if (type(bns) is not list):
        bns = [bns]

    def convert_to_nx(bn):
        """
        Converts a Bayesian network to a NetworkX DiGraph.
        """
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(bn.nodes())
        nx_graph.add_edges_from([(u, v) for u, v in bn.arcs()])
        return nx_graph

    # Convert Bayesian networks to NetworkX graphs
    nx_bns = [convert_to_nx(bn) for bn in bns]

    # Generate consistent node positions
    pos = nx.spring_layout(nx_bns[0], k=1)  # Compute layout for the first graph
    # pos = nx.random_layout(nx_bns[0])

    # Create subplots
    fig, axes = plt.subplots(1, len(nx_bns), figsize=(5*len(nx_bns), 5))
    if type(axes) not in [list, np.ndarray]:
        axes = [axes]
    # print(axes)
    
    if len(node_colors) != len(nx_bns):
        node_colors = [node_colors[0]] * len(nx_bns)

    for i in range(len(nx_bns)):
        nx.draw(
            nx_bns[i],
            pos,
            ax=axes[i],
            with_labels=with_labels,
            node_size=node_size,
            node_color=node_colors[i],
            font_size=font_size,
            font_weight=font_weight,
            arrowsize=arrowsize,
        )
        axes[i].set_title(f"Bayesian Network {i+1}")
    # Display the plots
    plt.tight_layout()
    plt.savefig("output/linear_compared_DAG.png", dpi=300)
    #plt.show()

    # Create subplots
    fig, axes = plt.subplots(1, len(nx_bns), figsize=(5*len(nx_bns), 5))
    if type(axes) not in [list, np.ndarray]:
        axes = [axes]
    for i in range(len(nx_bns)):
        sns.heatmap(nx.to_numpy_array(nx_bns[i]), ax=axes[i], 
                    # cmap="Blues", 
                    cbar=False
                    )
        axes[i].set_title(f"Adjacency Matrix {i+1}")
    plt.tight_layout()
    plt.savefig("output/linear_compared_adjacency_matrix.png", dpi=300)
    #plt.show()


def sample_data_from_bn(bn, num_samples):
    # df, likelihood = gum.generateSample(bn, num_samples)
    # # df = df[sorted(df.columns)]
    # df = df.astype(int)
    # return df, likelihood
    g=gum.BNDatabaseGenerator(bn)
    g.drawSamples(num_samples)
    return g.to_pandas().to_numpy(dtype=int)


def adjacency_matrix(bn):
    size = bn.size()
    adj_matrix = np.zeros((size, size), dtype=int)
    for node_id in range(size):
        for child_id in bn.children(node_id):
            adj_matrix[node_id, child_id] = 1
    return adj_matrix


def get_adj(bn,n=None):
    if n is None:
        n = bn.size()
    adj = np.zeros((n,n))
    for i in bn.arcs():
        adj[i[1],i[0]]=1
        adj[i[0],i[1]]=-1
    return adj


def get_ess_adj(bnEs, n):
    ess_adj = np.zeros((n,n))
    for i in bnEs.arcs():
        ess_adj[i[1],i[0]]=1
        ess_adj[i[0],i[1]]=-1
    for i in bnEs.edges():
        ess_adj[i[1],i[0]]=-1
        ess_adj[i[0],i[1]]=-1
    return ess_adj


def from_numpy_to_bn(B):
    # Create an empty Bayesian Network
    bn = gum.BayesNet()

    # Add nodes to the network
    num_nodes = B.shape[0]
    for i in range(num_nodes):
        bn.add(gum.LabelizedVariable(f"Node{i}", f"Node {i}", 2))  # Binary variables

    # Add edges based on the adjacency matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if B[i][j] == 1:  # Add edge if there's a 1 in the matrix
                bn.addArc(f"Node{i}", f"Node{j}")
    return bn


def generate_random_bn_like_DAGMA(n=10, sparsity=None, num_edges=None, ratio_arc=None, graph_type='ER'):
    if num_edges is None:
        assert (sparsity is not None) ^ (ratio_arc is not None), "Either sparsity or ratio_arc must be specified."
        if sparsity is not None:
            num_edges = int(sparsity*n*(n-1)/2)
        elif ratio_arc is not None:
            num_edges = int(ratio_arc*n)
    print(f"n={n}, num_edges={num_edges}, sparsity={sparsity}, ratio_arc={ratio_arc}")
    B = simulate_dag(n, num_edges, graph_type)
    # bn = from_numpy_to_bn(B)
    return B


def sample_data_from_bn_like_DAGMA(B, sem_type='gauss', w_ranges=((-2.0, -0.5), (0.5, 2.0)), nonlinear=False, N=1000,):
    W = simulate_parameter(B, w_ranges=w_ranges)
    kwargs = dict()
    f = simulate_linear_sem
    if nonlinear:
        f = simulate_nonlinear_sem
        sem_type = 'mlp'
    X = f(W, N, sem_type, noise_scale=10 if sem_type == 'gauss_mixture' else None, **kwargs)
    return X, W


def _sample_multi_mod_data(n, Ws, sem_func, sem_type='gauss', seed=None):
    rng = np.random.default_rng(seed=seed)
    nmod = Ws.shape[0]
    K = rng.choice(range(nmod), size=n, replace=True)
    Wp = np.zeros(shape=(n,n))
    for i,k in enumerate(K):
        Wp[i] = Ws[k].T[i]
    Wp = Wp.T
    x = sem_func(Wp, 1, sem_type,)[0]
    return x


def sample_multi_mod_data_from_bn_like_DAGMA(B, sem_type='gauss', w_ranges=((-2.0, -0.5), (0.5, 2.0)), nonlinear=False, N=1000, nmod=5, seed=None):
    rng = np.random.default_rng(seed=seed)
    n = B.shape[0]
    Ws = []
    for i in range(nmod):
        w_ranges = ((-2.*(i+1), -0.5*(i+1)), (0.5*(i+1), 2.*(i+1)))
        W = simulate_parameter(B, w_ranges=w_ranges)
        Ws.append(W)
    Ws = np.stack(Ws)
    f = simulate_linear_sem
    if nonlinear:
        f = simulate_nonlinear_sem
        sem_type = 'mlp'
    Xs = []
    pool = multiprocessing.Pool(os.cpu_count())
    jobs = []
    for _ in range(N):
        jobs.append(pool.apply_async(_sample_multi_mod_data, (n, Ws, f), dict(sem_type=sem_type, seed=seed)))
    pool.close()
    pool.join()
    for job in jobs:
        Xs.append(job.get())
    X = np.stack(Xs)
    return X, Ws