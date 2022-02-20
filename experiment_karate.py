import json
import operator
import os
import sys
from collections import defaultdict
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sparse
from sklearn.manifold import TSNE
from tqdm import tqdm

import maxent
from cne import ConditionalNetworkEmbedding
from utils import (compute_grad_approx, compute_grad_exact, from_cache,
                   to_cache, compute_robustness)

WORK_FOLDER = './'
DATA_CACHE_FILE = 'karate_data_synthetic.pkl'
CACHE_FOLDER = 'cache'
FIGURE_FOLDER = 'figures'
EXPERIMENT = 'karate'

if not os.path.exists(join(WORK_FOLDER, CACHE_FOLDER)):
    os.makedirs(join(WORK_FOLDER, CACHE_FOLDER))

if not os.path.exists(join(WORK_FOLDER, FIGURE_FOLDER)):
    os.makedirs(join(WORK_FOLDER, FIGURE_FOLDER))

def load_data(recompute=False):
    """
    load data.
    """
    path_to_cache_file = join(WORK_FOLDER, CACHE_FOLDER, DATA_CACHE_FILE)
    data = from_cache(path_to_cache_file)
    if data is not None:
        return data['A'], data['E']

    work_folder = 'data/karate'
    data_file = 'karate_0.edgelist'
    n = 34
    E = pd.read_csv(join(work_folder, data_file), header=None, delimiter=',').values.astype(int)
    A = sparse.csr_matrix((np.ones(E.shape[0]), (E[:,0], E[:,1])), shape=(n,n))
    A = (A + A.T).astype(bool)

    print('#nodes: {:d}'.format(A.shape[0]))

    # persist data
    data = {'A': A, 'E': E,}
    to_cache(path_to_cache_file, data)
    return data['A'], data['E']


def compute_cne_embedding(A, dim=2, s1=1, s2=2, lr=0.01, recompute=False):
    EMB_CACHE_FILE = '{:s}_cne_{:d}_synthetic.pkl'.format(EXPERIMENT, dim)
    path_to_cache_file = join(WORK_FOLDER, CACHE_FOLDER, EMB_CACHE_FILE)
    data = from_cache(path_to_cache_file)
    if data is not None and recompute is False:
        return data['cne']

    prior = maxent.BGDistr(A.tocsr())
    prior.fit()

    cne = ConditionalNetworkEmbedding(A, dim, s1, s2, prior)
    cne.fit(lr=lr)

    # persist data
    data = {
        'cne': cne
    }
    to_cache(path_to_cache_file, data)

    return data['cne']

def plot_graph(i, j, A, graph, X, grad, figure_file):
    fig = plt.figure(figsize=(6, 6))

    pos = X
    pos = pos - pos.mean(axis=0)

    A_i = A[i,:].todense()
    ids = np.where(A_i == 0)[1]
    grad[ids] = 0
    grad[i] = 0
    grad[j] = 0
    # edges
    nx.draw_networkx_edges(graph,pos,width=0.7,alpha=0.5)    
    nx.draw_networkx_edges(graph,pos, edgelist=[(i,j)], style='dashed', width=0.7,alpha=0.5)

    # nodes
    plt.scatter(pos[i, 0], pos[i, 1], 400, c='#66c2a5', marker='p', alpha=.8,
                label='node i')

    plt.scatter(pos[j, 0], pos[j, 1], 300, c='#66c2a5', marker='s', alpha=.8,
                label='node j')

    pos_mask = grad > 0
    plt.scatter(pos[pos_mask, 0], pos[pos_mask, 1], 350, c='#fc8d62',
                alpha=.8, label='node k, pos. effect', linestyle=':', edgecolors='0.2')

    neg_mask = grad < 0
    plt.scatter(pos[neg_mask, 0], pos[neg_mask, 1], 350, c='#8da0cb',
                alpha=.8, label='node k, neg. effect', linestyle='--', edgecolors='0.3')

    A_i[0,i] = 1
    A_i[0,j] = 1
    ids = np.where(A_i == 0)[1]
    # print(ids)
    plt.scatter(pos[ids, 0], pos[ids, 1], 350, 'k',
                alpha=0.3, label='other nodes')


    # labels
    nx.draw_networkx_labels(graph, pos=pos, font_size=10, font_weight='bold')


    scale = 1.05
    max_x = np.max(pos[:,0]) * scale
    min_x = np.min(pos[:,0]) * scale
    max_y = np.max(pos[:,1]) * scale
    min_y = np.min(pos[:,1]) * scale


    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.axis('off')
    legend = plt.legend(numpoints=1, labelspacing=1.5, frameon=False,
                        loc='lower right', fontsize=9)
    ax = plt.gca()

    plt.tight_layout()
    plt.savefig(join(WORK_FOLDER, FIGURE_FOLDER, figure_file))
    plt.close(fig)

def qualitative_experiment():
    i = 33
    j = 0
    k = i

    A, E = load_data()
    cne = compute_cne_embedding(A, dim=2, s1=1, s2=50, lr=0.1)
    X = cne.embedding
    cne_params = cne.parameters
    n = cne_params['n']
    s1 = cne_params['s1']
    s2 = cne_params['s2']
    post_P = np.vstack([cne.get_posterior_row(k) for k in range(n)])
    grad_exact,_ = compute_grad_exact(i, j, k, X, A.todense(), post_P, s1, s2)
    graph = nx.from_scipy_sparse_matrix(A)
    plot_graph(i, j, A, graph, X, grad_exact, 'intro_illustr_synthetic.pdf')

np.random.seed(0)
qualitative_experiment()
