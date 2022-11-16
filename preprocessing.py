'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import bernoulli
import torch


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # adj_normalized = adj_.tocoo()
    return sparse_to_tuple(adj_normalized)  # adj_normalized

# def preprocess_x(adj):
#     adj = sp.coo_matrix(adj)
#     adj_ = adj + sp.eye(adj.shape[0])
#     rowsum = np.array(adj_.sum(1))
#     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     adj_normalized = sparse_to_tuple(adj_normalized)
#     x = torch.sparse.FloatTensor(torch.LongTensor(adj_normalized[0].astype(float).T),
#                              torch.FloatTensor(adj_normalized[1].astype(float)),
#                              torch.Size(adj_normalized[2]))
#     return x

def preprocess_bow(bow):
    sums = bow.sum(1)
    normalized_bow = bow / sums
    normalized_bow = sp.csr_matrix(normalized_bow)
    return normalized_bow