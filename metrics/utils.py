from typing import Union
import os
from functools import partial
from itertools import product
import hashlib
from collections import deque
import networkx as nx
import torch
import torch.multiprocessing as mp
import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import typing
from torch.autograd import grad
from scipy.special import expit
import matplotlib.pyplot as plt
from tqdm import tqdm


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_hash(params_string: str) -> str:
    R"""
    Create hash from command line argument string. This is mainly for logging purposes.
    """
    hasher = hashlib.md5()
    hasher.update(params_string.encode('utf-8'))
    raw_hash =  hasher.hexdigest()
    hash_str = "{}".format(raw_hash)[:8]
    return hash_str


# source: https://www.geeksforgeeks.org/cartesian-product-of-any-number-of-sets/
def cartesianProduct(set_a, set_b): 
    result =[] 
    for i in range(0, len(set_a)): 
        for j in range(0, len(set_b)): 
  
            # for handling case having cartesian 
            # prodct first time of two sets 
            if type(set_a[i]) != list:          
                set_a[i] = [set_a[i]] 
                  
            # coping all the members 
            # of set_a to temp 
            temp = [num for num in set_a[i]] 
              
            # add member of set_b to  
            # temp to have cartesian product      
            temp.append(set_b[j])              
            result.append(temp)   
              
    return result 


# Function to do a cartesian  
# product of N sets  
def Cartesian(list_a, n): 
      
    # result of cartesian product 
    # of all the sets taken two at a time 
    if len(list_a)==0:
        return []
    else:
        temp = list_a[0] 

        # do product of N sets  
        for i in range(1, n): 
            temp = cartesianProduct(temp, list_a[i]) 
    return temp


# edit CPTs
def create_CPT(bn,var_name,no_of_states_dict,option,vec=None):
    if option=='random':
        bn.generateCPT(var_name)

    elif option=='logistic_binary':
        #print(var_name)
        parent_names=bn.cpt(var_name).var_names
        for j in parent_names:
            assert no_of_states_dict[j]==2, "logistic_binary can only be used with binary variables"
        parent_names.remove(var_name)
        #print(parent_names)
        assert(len(parent_names)+1== len(vec)), "Length of the vector of coefficients mis matched with the number of parents"
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        #print(parent_states)
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            my_dist=[vec[k]*int(j[k]) for k in range(len(parent_names))]
            logit=np.sum(np.array(my_dist))+vec[-1]
            #print(logit)
            bn.cpt(var_name)[my_dict] = np.array([expit(logit),1-expit(logit)])

    elif option=='deterministic':
        alpha=np.zeros((no_of_states_dict[var_name],))
        #print(no_of_states_dict[var_name])
        alpha[1]=1
        #print(alpha)

        parent_names=bn.cpt(var_name).var_names
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            #print(j)
            alpha_shifted=np.roll(alpha,counter)
            #print({k:1 for k in range(len(parent_names))})        
            #print({parent_names[k]:j for k in range(len(parent_names))})
            #print(parent_names)
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            #print(my_dict)
            my_dist=alpha_shifted
            #print(my_dist)
            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1
    elif option=='Dirichlet':
        alpha=np.ones((no_of_states_dict[var_name],))
        #print(no_of_states_dict[var_name])
        #print(alpha)

        parent_names=bn.cpt(var_name).var_names
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            my_dist=list(np.random.dirichlet(tuple(alpha), 1)[0])
            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1
    elif option=='Meek':
        base=1./np.arange(1,no_of_states_dict[var_name]+1)
        base=base/np.sum(base)
        # equivalent sample size = sum of ai's in Dirichlet
        alpha=10*base
        parent_names=bn.cpt(var_name).var_names
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        #print(parent_names)
        #print(parent_states)
        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            #print(j)
            alpha_shifted=np.roll(alpha,counter)
            # alpha_shifted=np.roll(alpha,np.random.choice(no_of_states[i],1))        
            #print({k:1 for k in range(len(parent_names))})        
            #print({parent_names[k]:j for k in range(len(parent_names))})
            #print(parent_names)
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            #print(my_dict)

            my_dist=list(np.random.dirichlet(tuple(alpha_shifted), 1)[0])

            #my_dist=alpha_shifted

            #print(my_dist)
            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1
    elif option=='reverseDeterministic':
        #print(var_name)
        parent_names=bn.cpt(var_name).var_names
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        #print(parent_states)
        M=np.zeros([len(parent_states),no_of_states_dict[var_name]])
        ind=np.random.choice(len(parent_states),no_of_states_dict[var_name])
        for i in range(no_of_states_dict[var_name]):
            M[ind[i],i]=1
            M=M/np.repeat(np.reshape(np.sum(M,1),[-1,1]),no_of_states_dict[var_name],axis=1)
            for j in parent_states:
                if not (isinstance(j,list)):
                    j=[j]
                my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
                my_dist=M[j,:]
                #print(logit)
                bn.cpt(var_name)[my_dict] = my_dist


def nx_d_separation(G: nx.DiGraph, X: set, Y: set, Z: set) -> bool:
    """
    Returns the ground-truth d-separation of X and Y given Z in G.
    This is a wrapper around NetworkX. Decides what function to call because apparently NetworkX has changed its API 
        after version 3.3.
    """
    # Get the version of NetworkX
    version = nx.__version__
    # If <= 3.2.1, use the old API
    if version <= '3.2.1':
        return nx.d_separated(G, X, Y, Z)
    # If > 3.2.1, use the new API
    else:
        return nx.is_d_separator(G, X, Y, Z)


def is_dag(W: np.ndarray) -> bool:
    """
    Returns ``True`` if ``W`` is a DAG, ``False`` otherwise.
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def true_d_sep_sp(G, n):
    # Single-process version of true_d_sep
    # Obtain ground-truth 0-th and 1-st order d-separation matrices
    true_d_sep_0 = torch.zeros(n, n)
    total_d_sep_0 = 0
    # print("Computing the ground-truth 0th-order d-separation statements...")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if i and j are d-separated
            nx_d_sep_0 = nx_d_separation(G, {i}, {j}, set())
            total_d_sep_0 += nx_d_sep_0
            true_d_sep_0[i, j] = nx_d_sep_0
    
    true_d_sep_1 = torch.zeros(n, n, n)   # k, i, j
    total_d_sep_1 = 0
    # print("Computing the ground-truth 1st-order d-separation statements...")
    for k in range(n):
            for i in range(n):
                for j in range(i+1, n):
                    # Avoid degenerate cases when i == k or j == k
                    if i == k or j == k:
                        true_d_sep_1[k, i, j] = 1  # i and j are always conditionally independent given i or j
                        continue
                    # Check if i and j are d-separated conditioned on k
                    nx_d_sep_1 = nx_d_separation(G, {i}, {j}, {k})
                    total_d_sep_1 += nx_d_sep_1
                    true_d_sep_1[k, i, j] = nx_d_sep_1
                    true_d_sep_1[k, j, i] = nx_d_sep_1  # Symmetric
            # Fix degenerate cases when i == k or j == k
            # Follow the convention that they should be considered always d-separated (independent)
            true_d_sep_1[k, k, :] = 1
            true_d_sep_1[k, :, k] = 1

    # print(f"Total 0-th order d-separation: {total_d_sep_0}")
    # print(f"Total 1-st order d-separation: {total_d_sep_1}")
    true_d_sep = torch.cat([true_d_sep_0.unsqueeze(0), true_d_sep_1], dim=0)  # (n+1, n, n)

    return true_d_sep


# Define worker functions within this function scope
def compute_d_sep_0(i, j, G):
    if i == j:
        return 0
    return nx_d_separation(G, {i}, {j}, set())

def compute_d_sep_1(k, i, j, G):
    if i == k or j == k:
        return 1
    return nx_d_separation(G, {i}, {j}, {k})


def true_d_sep(G, n):
    """
    Compute all 0th and 1st order d-separation statements from a NetworkX graph.
    Uses multiprocessing when n > 10.
    
    Args:
        G: NetworkX DiGraph
        n: Number of nodes
        
    Returns:
        torch.Tensor: (n+1, n, n) tensor of d-separation statements
    """
    # Initialize tensors
    true_d_sep_0 = torch.zeros(n, n)
    true_d_sep_1 = torch.zeros(n, n, n)  # k, i, j
    
    # Determine if we should use multiprocessing
    use_mp = n > 20
    
    if use_mp:
        try:
            # Get number of available cores
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:  # For non-Linux platforms
            num_cores = mp.cpu_count()
        
        # Limit to a reasonable number
        num_cores = min(num_cores, 16)
        
        # Create a process pool
        with mp.Pool(processes=num_cores) as pool:
            # 0th order calculations
            args_0 = [(i, j, G) for i in range(n) for j in range(n) if i != j]
            results_0 = pool.starmap(compute_d_sep_0, args_0)
            
            # Map results back to the tensor
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        true_d_sep_0[i, j] = results_0[idx]
                        idx += 1
            
            # 1st order calculations
            args_1 = []
            for k in range(n):
                for i in range(n):
                    for j in range(i+1, n):
                        if i != k and j != k:
                            args_1.append((k, i, j, G))
            
            results_1 = pool.starmap(compute_d_sep_1, args_1)
            
            # Map results back to the tensor
            idx = 0
            for k in range(n):
                for i in range(n):
                    for j in range(i+1, n):
                        if i != k and j != k:
                            true_d_sep_1[k, i, j] = results_1[idx]
                            true_d_sep_1[k, j, i] = results_1[idx]  # Symmetric
                            idx += 1
    else:
        # Original sequential computation for small graphs
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                true_d_sep_0[i, j] = nx_d_separation(G, {i}, {j}, set())
        
        for k in range(n):
            for i in range(n):
                for j in range(i+1, n):
                    # Avoid degenerate cases
                    if i == k or j == k:
                        true_d_sep_1[k, i, j] = 1
                        true_d_sep_1[k, j, i] = 1
                        continue
                    # Check d-separation
                    nx_d_sep_1 = nx_d_separation(G, {i}, {j}, {k})
                    true_d_sep_1[k, i, j] = nx_d_sep_1
                    true_d_sep_1[k, j, i] = nx_d_sep_1  # Symmetric
    
    # Handle degenerate cases for both parallel and sequential paths
    for k in range(n):
        true_d_sep_1[k, k, :] = 1
        true_d_sep_1[k, :, k] = 1
    
    # Combine into final tensor
    true_d_sep = torch.cat([true_d_sep_0.unsqueeze(0), true_d_sep_1], dim=0)  # (n+1, n, n)
    
    return true_d_sep


def simulate_dag(d: int, s0: int, graph_type: str) -> np.ndarray:
    r"""
    Simulate random DAG with some expected number of edges.

    Parameters
    ----------
    d : int
        num of nodes
    s0 : int
        expected num of edges
    graph_type : str
        One of ``["ER", "SF", "BP"]``
    
    Returns
    -------
    numpy.ndarray
        :math:`(d, d)` binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == 'Fully':
        B = np.triu(np.ones((d,d)), 1)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B: np.ndarray, 
                       w_ranges: typing.List[typing.Tuple[float,float]]=((-2.0, -0.5), (0.5, 2.0)),
                       ) -> np.ndarray:
    r"""
    Simulate SEM parameters for a DAG.

    Parameters
    ----------
    B : np.ndarray
        :math:`[d, d]` binary adj matrix of DAG.
    w_ranges : typing.List[typing.Tuple[float,float]], optional
        disjoint weight ranges, by default :math:`((-2.0, -0.5), (0.5, 2.0))`.

    Returns
    -------
    np.ndarray
        :math:`[d, d]` weighted adj matrix of DAG.
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W: np.ndarray, 
                        n: int, 
                        sem_type: str, 
                        noise_scale: typing.Optional[typing.Union[float,typing.List[float]]] = None,
                        ) -> np.ndarray:
    r"""
    Simulate samples from linear SEM with specified type of noise.
    For ``uniform``, noise :math:`z \sim \mathrm{uniform}(-a, a)`, where :math:`a` is the ``noise_scale``.
    
    Parameters
    ----------
    W : np.ndarray
        :math:`[d, d]` weighted adj matrix of DAG.
    n : int
        num of samples. When ``n=inf`` mimics the population risk, only for Gaussian noise.
    sem_type : str
        ``gauss``, ``exp``, ``gumbel``, ``uniform``, ``logistic``, ``poisson``
    noise_scale : typing.Optional[typing.Union[float,typing.List[float]]], optional
        scale parameter of the additive noises. If ``None``, all noises have scale 1. Default: ``None``.

    Returns
    -------
    np.ndarray
        :math:`[n, d]` sample matrix, :math:`[d, d]` if ``n=inf``.
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or have length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B:np.ndarray, 
                           n: int, 
                           sem_type: str, 
                           noise_scale: typing.Optional[typing.Union[float,typing.List[float]]] = None,
                           ) -> np.ndarray:
    r"""
    Simulate samples from nonlinear SEM.

    Parameters
    ----------
    B : np.ndarray
        :math:`[d, d]` binary adj matrix of DAG.
    n : int
        num of samples
    sem_type : str
        ``mlp``, ``mim``, ``gp``, ``gp-add``
    noise_scale : typing.Optional[typing.Union[float,typing.List[float]]], optional
        scale parameter of the additive noises. If ``None``, all noises have scale 1. Default: ``None``.

    Returns
    -------
    np.ndarray
        :math:`[n, d]` sample matrix.
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    r"""
    Compute various accuracy metrics for B_est.

    | true positive = predicted association exists in condition in correct direction
    | reverse = predicted association exists in condition in opposite direction
    | false positive = predicted association does not exist in condition
    
    Parameters
    ----------
    B_true : np.ndarray
        :math:`[d, d]` ground truth graph, :math:`\{0, 1\}`.
    B_est : np.ndarray
        :math:`[d, d]` estimate, :math:`\{0, 1, -1\}`, -1 is undirected edge in CPDAG.

    Returns
    -------
    dict
        | fdr: (reverse + false positive) / prediction positive
        | tpr: (true positive) / condition positive
        | fpr: (reverse + false positive) / condition negative
        | shd: undirected extra + undirected missing + reverse
        | nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}


# Utility function for debugging gradients
def register_hook(x, x_name):
    # Register a hook to print the gradient information of the given tensor
    assert x.requires_grad
    x.register_hook(lambda grad: print(f"{x_name} - Gradient norm: {grad.norm()}, Gradient abs max: {grad.abs().max()}, Gradient abs min: {grad.abs().min()}"))

def get_grad(y, x):
    # Get the gradient of y w.r.t. x
    y = y.sum()
    grad_x = grad(y, x, retain_graph=True)[0]  # Do not destroy the computation graph
    return grad_x

def get_x_delta(y, x):
    # Get negative gradient of y w.r.t. x, i.e., the x delta that decreases y
    grad_x = get_grad(y, x)
    return -grad_x


def draw_dag(adj: Union[torch.Tensor, np.ndarray], fig_path="debug_dag.png"):
    """
    Draw the DAG using NetworkX and save it to a file.
    """
    assert isinstance(adj, (torch.Tensor, np.ndarray)), "Input must be a torch.Tensor or numpy.ndarray"
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    adj = adj.astype(int)  # Ensure the adjacency matrix is of type int
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)  # Circular layout for better visualization
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")
    plt.title("Directed Acyclic Graph (DAG)")
    plt.savefig(fig_path)
    plt.close()


def graph2dag(adj: np.ndarray) -> np.ndarray:
    """
    Convert a graph adjacency matrix to a DAG adjacency matrix via heuristics.
    """
    assert isinstance(adj, np.ndarray), "Input must be a numpy.ndarray"
    assert adj.ndim == 2 and adj.shape[0] == adj.shape[1], "Input must be a square matrix"
    
    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    
    # Check if the graph is already a DAG
    if nx.is_directed_acyclic_graph(G):
        return adj
    
    # If not, convert it to a DAG by removing edges from minimum feedback arc set
    # Convert NetworkX graph to igraph format
    edges = list(G.edges())
    
    # Create igraph Graph
    g = ig.Graph(directed=True)
    g.add_vertices(adj.shape[0])  # Add vertices based on matrix dimensions
    g.add_edges(edges)
    
    # Find feedback arc set (edges to remove)
    fas = g.feedback_arc_set(weights=None)
    feedback_edges = [edges[i] for i in fas]

    # Remove feedback edges from the original graph
    G.remove_edges_from(feedback_edges)
    dag_adj = nx.to_numpy_array(G, dtype=int)
    return dag_adj


def is_d_separated_pdag(pdag, x, y, z):
    """
    Check d-separation in PDAG using causal-learn's adjacency conventions.

    Args:
        pdag (np.ndarray): PDAG adjacency matrix.
        x (int): Node X.
        y (int): Node Y.
        z (set): Set of nodes Z.
    Returns:
        bool: True if X and Y are d-separated given Z, False otherwise.
    """
    n = pdag.shape[0]
    visited = set()
    queue = deque()
    
    # Initialize with X's neighbors
    for neighbor in get_adjacent(pdag, x):
        if is_directed_edge(pdag, x, neighbor):  # x → neighbor
            queue.append((neighbor, x, False, 'child'))
        elif is_directed_edge(pdag, neighbor, x):  # x ← neighbor
            queue.append((neighbor, x, False, 'parent'))
        else:  # x − neighbor (undirected)
            queue.append((neighbor, x, False, 'child'))
            queue.append((x, neighbor, False, 'parent'))

    while queue:
        current, prev, is_collider, direction = queue.popleft()
        
        if (current, prev, is_collider) in visited:
            continue
        visited.add((current, prev, is_collider))
        
        # Block non-colliders in Z
        if not is_collider and current in z:
            continue
            
        # Path reaches Y
        if current == y:
            if not is_collider:
                return False
            if is_collider and has_descendant_in_z(pdag, current, z):
                return False
            
        # Process neighbors
        for next_node in get_adjacent(pdag, current):
            if next_node == prev:
                continue
                
            # Edge type detection
            new_collider = is_collider
            new_dir = None
            
            if is_directed_edge(pdag, current, next_node):  # current → next_node
                if direction == 'parent':
                    new_collider = True
                new_dir = 'child'
            elif is_directed_edge(pdag, next_node, current):  # current ← next_node
                new_dir = 'parent'
            else:  # undirected
                new_dir = 'child' if direction == 'parent' else 'parent'
            
            if new_dir:
                queue.append((next_node, current, new_collider, new_dir))

    return True

def get_adjacent(pdag, node):
    """Get all nodes adjacent to 'node' (any edge connection)"""
    return np.where((pdag[node] != 0) | (pdag[:, node] != 0))[0].tolist()

def is_directed_edge(pdag, i, j):
    """Check for i → j directed edge"""
    return pdag[j, i] == 1 and pdag[i, j] == -1

def has_descendant_in_z(pdag, node, z):
    """Check descendants via directed edges"""
    visited = set()
    queue = deque([node])
    
    while queue:
        current = queue.popleft()
        if current in z:
            return True
        visited.add(current)
        
        # Get children via i → j edges
        children = [j for j in range(pdag.shape[0]) 
                   if is_directed_edge(pdag, current, j)]
        
        queue.extend([c for c in children if c not in visited])
    
    return False


def true_d_sep_pdag_sp(pdag, n):
    """
    Get all 0th- and 1st-order d-separation statements from a PDAG.

    Args:
        pdag (np.ndarray): PDAG adjacency matrix.
        n (int): Number of nodes in the PDAG.
    Returns:
        d_sep_matrix: np.ndarray, of shape (n+1, n, n)
            - 0th-order d-separation statements (first slice)
            - 1st-order d-separation statements (remaining slices)
    """
    d_sep_matrix = np.zeros((n+1, n, n), dtype=bool)
    
    # 0th-order d-separation
    for i in range(n):
        for j in range(n):
            if i != j:
                d_sep_matrix[0, i, j] = is_d_separated_pdag(pdag, i, j, set())
    
    # 1st-order d-separation
    for k in range(n):
        for i in range(n):
            for j in range(i+1, n):
                if i != k and j != k:
                    d_sep_matrix[k+1, i, j] = is_d_separated_pdag(pdag, i, j, {k})
                    d_sep_matrix[k+1, j, i] = d_sep_matrix[k+1, i, j]  # Symmetric
                # Fix degenerate cases when i == k or j == k
                # Follow the convention that they should be considered always d-separated (independent)
        d_sep_matrix[k+1, k, :] = True
        d_sep_matrix[k+1, :, k] = True
    
    return d_sep_matrix.astype(int)


def true_d_sep_pdag(pdag, n):
    """
    Multi-process version of true_d_sep_pdag.
    """
    # Initialize tensors
    d_sep_matrix = np.zeros((n+1, n, n), dtype=bool)
    
    # Determine if we should use multiprocessing
    use_mp = n > 20
    
    if use_mp:
        try:
            # Get number of available cores
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:  # For non-Linux platforms
            num_cores = mp.cpu_count()
        
        # Limit to a reasonable number
        num_cores = min(num_cores, 16)
        
        # Create a process pool
        with mp.Pool(processes=num_cores) as pool:
            # 0th order calculations
            args_0 = [(pdag, i, j, {}) for i in range(n) for j in range(n) if i != j]
            results_0 = pool.starmap(is_d_separated_pdag, args_0)
            
            # Map results back to the tensor
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d_sep_matrix[0, i, j] = results_0[idx]
                        idx += 1
            
            # 1st order calculations
            args_1 = []
            for k in range(n):
                for i in range(n):
                    for j in range(i+1, n):
                        if i != k and j != k:
                            args_1.append((pdag, i, j, {k}))
            
            results_1 = pool.starmap(is_d_separated_pdag, args_1)
            
            # Map results back to the tensor
            idx = 0
            for k in range(n):
                for i in range(n):
                    for j in range(i+1, n):
                        if i != k and j != k:
                            d_sep_matrix[k+1, i, j] = results_1[idx]
                            d_sep_matrix[k+1, j, i] = results_1[idx]  # Symmetric
                            idx += 1
    else:
        # Original sequential computation for small graphs
        d_sep_matrix = true_d_sep_pdag_sp(pdag, n)
        
    return d_sep_matrix.astype(int)


if __name__ == "__main__":
    # Unit test of true_d_sep: Check that it produce the same results as true_d_sep_sp
    import time
    from tqdm import tqdm
    
    n_nodes = 50
    n_edges = 50
    num_graphs = 3

    # Generate num_graphs random DAGs
    dags_adj = [simulate_dag(n_nodes, n_edges, 'ER') for _ in range(num_graphs)]
    dags = [nx.from_numpy_array(adj, create_using=nx.DiGraph) for adj in dags_adj]
    
    # Measure time and get results for true_d_sep_sp
    single_process_results = []
    start_time = time.time()
    for dag in tqdm(dags, desc="Computing true_d_sep_sp"):
        result = true_d_sep_sp(dag, n_nodes)
        single_process_results.append(result)
    single_process_time = time.time() - start_time
    print(f"Single-process time: {single_process_time:.4f} seconds")

    # Measure time and get results for true_d_sep
    multi_thread_results = []
    start_time = time.time()
    for dag in tqdm(dags, desc="Computing true_d_sep"):
        result = true_d_sep(dag, n_nodes)
        multi_thread_results.append(result)
    multi_thread_time = time.time() - start_time
    print(f"Multi-threaded time: {multi_thread_time:.4f} seconds")

    # Compare results
    for i in range(num_graphs):
        assert np.array_equal(single_process_results[i], multi_thread_results[i]), "Results do not match!"

    pass