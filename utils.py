import json
import logging
import operator
import os
import pickle
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sparse
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def recommend(i, A, P_i):
    A_i = A[i, ].todense() == 0
    A_i[0, i] = False
    ids_neg = np.where(A_i)[1]
    P_i_neg = P_i[ids_neg]
    return sorted(zip(ids_neg, P_i_neg), key=operator.itemgetter(1), reverse=True)

def from_cache(path_to_cache_file):
    if os.path.exists(path_to_cache_file):
        result = []
        with open(path_to_cache_file, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return None

def to_cache(path_to_cache_file, data):
    with open(path_to_cache_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def compute_grad_approx(X_i, j, X, A_i, P_i, s1, s2):
    gamma = 1/s1**2 - 1/s2**2
    d = X.shape[1]
    X_j = X[j, :]
    ratio = np.tile(P_i*(1-P_i), (d,1))
    # J = 1/gamma*np.eye(d)*(P_i - A_i).sum() - ((X_i - X).T*ratio).dot(X_i - X)
    J = - ((X_i - X).T*ratio).dot(X_i - X)
    P_ij_X = -(X_i - X_j)*P_i[j]*(1-P_i[j])
    F_a_ik = (X - X_i)
    grad = P_ij_X[:,None].T.dot(np.linalg.solve(-J, F_a_ik.T))
    return grad.squeeze()

def compute_grad_exact(i, j, k, X, A, P, s1, s2, invH=None):
    gamma = 1/s1**2 - 1/s2**2
    n, d = X.shape

    if invH is None:
        print('Computing inversed Hessian...')
        H = np.zeros((n*d, n*d))
        for hi in tqdm(range(n)):
            H_ii = np.zeros((d, d))
            for hj in range(n):
                if hj != hi:
                    diff = (X[hi,:] - X[hj,:])[:,None]
                    # H_ij = -1./gamma*np.eye(d)*(P[hi,hj]-A[hi,hj]) + diff.dot(diff.T)*P[hi,hj]*(1-P[hi,hj])
                    H_ij = diff.dot(diff.T)*P[hi,hj]*(1-P[hi,hj])
                    H[hi*d:(hi+1)*d, hj*d:(hj+1)*d] = H_ij
                    H_ii -= H_ij
            H[hi*d:(hi+1)*d, hi*d:(hi+1)*d] = H_ii
        invH = np.linalg.pinv(-H)
        print('done.')

    P_X_ij = np.zeros((n*d,1))
    P_X_ij[i*d:(i+1)*d,:] = -((X[i,:] - X[j,:])*P[i,j]*(1-P[i,j]))[:,None]
    P_X_ij[j*d:(j+1)*d,:] = -((X[j,:] - X[i,:])*P[i,j]*(1-P[i,j]))[:,None]

    grad = np.zeros(n)
    l = 0
    for l in range(n):
        F_X_kl = np.zeros((n*d,1))
        F_X_kl[k*d:(k+1)*d] = (X[l] - X[k,:])[:,None]
        F_X_kl[l*d:(l+1)*d] = (X[k] - X[l,:])[:,None]
#         grad[l] = P_X_ij.T.dot(np.linalg.solve(-H, F_X_kl))
        grad[l] = P_X_ij.T.dot(invH.dot(F_X_kl))

    return grad, invH

def compute_robustness(X, A, P, s1, s2, invH=None):
    gamma = 1/s1**2 - 1/s2**2
    n, d = X.shape

    if invH is None:
        print('Computing inversed Hessian...')
        H = np.zeros((n*d, n*d))
        for hi in tqdm(range(n)):
            H_ii = np.zeros((d, d))
            for hj in range(n):
                if hj != hi:
                    diff = (X[hi,:] - X[hj,:])[:,None]
                    # H_ij = -1./gamma*np.eye(d)*(P[hi,hj]-A[hi,hj]) + diff.dot(diff.T)*P[hi,hj]*(1-P[hi,hj])
                    H_ij = diff.dot(diff.T)*P[hi,hj]*(1-P[hi,hj])
                    H[hi*d:(hi+1)*d, hj*d:(hj+1)*d] = H_ij
                    H_ii -= H_ij
            H[hi*d:(hi+1)*d, hi*d:(hi+1)*d] = H_ii
        invH = np.linalg.pinv(-H)
        print('done.')

    E = []
    is_edge = []
    result = []
    for k in range(n-1):
        for l in range(k+1, n):
            F_X_kl = np.zeros((n*d,1))
            F_X_kl[k*d:(k+1)*d] = (X[l] - X[k,:])[:,None]
            F_X_kl[l*d:(l+1)*d] = (X[k] - X[l,:])[:,None]

            E.append((k,l))
            is_edge.append(A[k,l])
            result.append(np.linalg.norm(invH.dot(F_X_kl)))
    return zip(E, is_edge, result)

def from_csr_matrix_to_edgelist(csr_A):
    csr_A = sparse.csr_matrix(csr_A)
    t_list = csr_A.indices
    h_list = np.zeros_like(t_list).astype(int)
    for i in range(csr_A.shape[0]):
        h_list[csr_A.indptr[i]:csr_A.indptr[i+1]] = i
    return np.vstack((h_list, t_list)).T

def split_mst(A):
    mst_A = sparse.csgraph.minimum_spanning_tree(A)

    mst_A = (mst_A + mst_A.T).astype(bool)
    rest_A = A - mst_A

    mst_E = from_csr_matrix_to_edgelist(sparse.triu(mst_A, 1))
    rest_E = from_csr_matrix_to_edgelist(sparse.triu(rest_A, 1))
    np.random.shuffle(rest_E)
    return mst_E, rest_E

def sample_neg_edges(A, n_edges):
    n_nodes = A.shape[0]
    portion = 1.5
    while True:
        sample_E = np.random.randint(n_nodes, size=(int(portion*n_edges), 2))
        sample_A  = sparse.csr_matrix((np.ones(len(sample_E)),
                                      (sample_E[:,0], sample_E[:,1])),
                                      shape=(n_nodes,n_nodes)).astype(bool)
        neg_A = sparse.triu(sample_A.astype(int) - A.astype(int), 1) > 0
        if np.sum(neg_A) > n_edges:
            break
        else:
            portion += 0.5
    neg_E = from_csr_matrix_to_edgelist(neg_A)
    np.random.shuffle(neg_E)
    return neg_E[:n_edges]

def split_pos_edges(A, cut_off):
    E_a, E_b = split_mst(A)
    split_E = np.vstack((E_a, E_b[:cut_off-len(E_a)]))
    rest_E = E_b[cut_off-len(E_a):]
    return split_E, rest_E

def split_neg_edges(A, cut_off):
    E = sample_neg_edges(A, int(np.sum(A)/2))
    return E[:cut_off], E[cut_off:]

def split_edges(A, cut_off):
    pos_E_a, pos_E_b = split_pos_edges(A, cut_off)
    neg_E_a, neg_E_b = split_neg_edges(A, cut_off)
    return pos_E_a, neg_E_a, pos_E_b, neg_E_b

def construct_adj_matrix(E, n):
    E = np.array(E)
    A  = sparse.csr_matrix((np.ones(len(E)), (E[:,0], E[:,1])),
                           shape=(n,n)).astype(bool)
    A = (A + A.T).astype(bool)
    return A

def label_edges(E, label):
    return np.hstack((E, np.tile(label, (len(E),1))))

def compute_tr_val_split(A, portion):
    cutt_off = int(np.sum(sparse.triu(A, 1))*portion)
    tr_pos_E, tr_neg_E, val_pos_E, val_neg_E = split_edges(A, cutt_off)

    tr_A = construct_adj_matrix(tr_pos_E, A.shape[0])
    tr_E = np.vstack((label_edges(tr_pos_E, 1),
                      label_edges(tr_neg_E, 0)))
    val_E = np.vstack((label_edges(val_pos_E, 1),
                      label_edges(val_neg_E, 0)))
    return tr_A, tr_E, val_E

def validate(optimizer, predict, A, tr_E, val_E, prior_dist, d, s1, s2):
    X0 = np.random.randn(A.shape[0], d)
    emb = optimizer(X0, A, prior_dist, s1, s2)

    tr_pred = predict(tr_E[:,:2], emb, prior_dist, s1, s2)
    tr_true = tr_E[:,2]

    val_pred = predict(val_E[:,:2], emb, prior_dist, s1, s2)
    val_true = val_E[:,2]

    tr_auc = roc_auc_score(tr_true, tr_pred)
    val_auc = roc_auc_score(val_true, val_pred)

    return val_auc
