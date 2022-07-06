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
    return sparse_to_tuple(adj_normalized)

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

###################### simulated data according to LPCM ###################
def create_simuA(N, K, delta):
    mu1 = [0, 0]
    mu2 = [delta * 1.5, delta * 1.5]
    mu3 = [-1.5 * delta, delta * 1.5]
    z_mu = np.concatenate((mu1,mu2,mu3), axis=0)

    sigma1 = [[0.1, 0],[0, 0.1]]
    sigma2 = [[0.3, 0],[0, 0.3]]
    sigma3 = [[0.1, 0],[0, 0.1]]
    z_log_sigma = np.concatenate((sigma1,sigma2,sigma3), axis=0)

    x1 = np.random.multivariate_normal(mu1, sigma1, N//K)
    x2 = np.random.multivariate_normal(mu2, sigma2, N//K)
    x3 = np.random.multivariate_normal(mu3, sigma3, N-2*(N//K))

    # import matplotlib.pyplot as plt
    # f, ax = plt.subplots(1,figsize=(8,8))
    # ax.scatter(x1[:,0], x1[:,1], color = '#7294d4')
    # ax.scatter(x2[:,0], x2[:,1], color = '#fdc765')
    # ax.scatter(x3[:,0], x3[:,1], color = '#869f82')
    # # ax.scatter(x4[:,0], x4[:,1], color = 'y')
    # # ax.scatter(x5[:,0], x5[:,1], color = 'purple')
    # ax.set_title("Original Embeddings of Scenario A (Delta=0.5)", fontsize=18)
    # plt.show()

    X = np.concatenate((x1,x2,x3), axis=0)
    # np.savetxt('emb_3clusters.txt', X)
    # np.savetxt('mu_3clusters.txt', z_mu)
    # np.savetxt('cov_3clusters.txt', z_log_sigma)
    Label1 = np.repeat(0, N//K)
    Label2 = np.repeat(1, N//K)
    Label3 = np.repeat(2, N-2*(N//K))
    Label = np.concatenate((Label1,Label2,Label3), axis=0)

    dst = pdist(X, 'euclidean')
    dst = squareform(dst)

    alpha = 0.2
    from scipy.special import expit
    from scipy.stats import bernoulli
    A = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            prob = expit(alpha - dst[i,j])
            A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)

    # np.savetxt('adj_simuA_3clusters.txt', A)
    # np.savetxt('label_simuA_3clusters.txt', Label)
    # f.savefig("C:/Users/Dingge/Desktop/results/emb_orig_A.pdf", bbox_inches='tight')

    return A, Label

###################### simulated data according to SBM ###################
def create_simuB(N, K, delta):
    Pi = np.zeros((K, K))
    b = 0.25
    c = b
    a = 0.01 + (1-delta) * (b-0.01)

    Pi[0,0] = a
    Pi[0,1] = b
    Pi[0,2] = b
    Pi[1,0] = b
    Pi[2,0] = b
    Pi[1,1] = c
    Pi[1,2] = a
    Pi[2,1] = a
    Pi[2,2] = c

    Rho = [0.1, 0.45, 0.45]
    # N=5
    c = np.random.multinomial(1, Rho, size=N)
    c = np.argmax(c, axis=1)


    from scipy.stats import bernoulli
    A = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            prob = Pi[c[i], c[j]]
            A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)

    label = []
    for idx in range(len(c)):
        if c[idx] == 0:
            label.append('#7294d4')
        elif c[idx] == 1:
            label.append('#fdc765')
        # elif labels[idx] == 2:
        #     labelC.append('yellow')
        # elif labels[idx] == 3:
        #     labelC.append('purple')
        else:
            label.append('#869f82')

    # np.savetxt('adj_simuB_3clusters.txt', A)
    # np.savetxt('label_simuB_3clusters.txt', c)
    print('Delta is................: ', delta)
    print('Clusters='+str(K))

    return A, c

###################### simulated data according to circle structure ###################
def create_simuC(N, K):

    x = np.random.uniform(-1,1,N//K)
    c = np.random.multinomial(1, [0.5,0.5], size=N//K)
    c = np.argmax(c, axis=1)
    y = np.sqrt(1 - x**2) + np.random.normal(0,0.1,N//K)
    y[c==1] = -y[c==1]

    x2 = np.random.uniform(-5,5,N//K)
    c = np.random.multinomial(1, [0.5,0.5], size=N//K)
    c = np.argmax(c, axis=1)
    y2 = np.sqrt(25-x2**2) + np.random.normal(0,0.1,N//K)
    y2[c==1] = -y2[c==1]

    x3 = np.random.uniform(-10,10,N-2*(N//K))
    c = np.random.multinomial(1, [0.5,0.5], size=N-2*(N//K))
    c = np.argmax(c, axis=1)
    y3 = np.sqrt(100-x3**2) + np.random.normal(0,0.1,N-2*(N//K))
    y3[c==1] = -y3[c==1]

    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, figsize=(8, 8))
    ax.scatter(x, y, color='#7294d4')
    ax.scatter(x2, y2, color='#fdc765')
    ax.scatter(x3, y3, color='#869f82')
    ax.set_title("Original Embeddings of Scenario C", fontsize=18)
    # f.savefig("C:/Users/Dingge/Desktop/results/emb_orig_C.pdf", bbox_inches='tight')

    K1 = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)
    K2 = np.concatenate((x2.reshape(-1,1),y2.reshape(-1,1)), axis=1)
    K3 = np.concatenate((x3.reshape(-1,1),y3.reshape(-1,1)), axis=1)

    C= np.concatenate((K1,K2,K3), axis=0)
    # np.savetxt('emb_3clusters.txt', K)

    Label1 = np.repeat(0, N//K)
    Label2 = np.repeat(1, N//K)
    Label3 = np.repeat(2, N-2*(N//K))
    Label = np.concatenate((Label1,Label2,Label3), axis=0)

    dst = pdist(C, 'euclidean')
    dst = squareform(dst)

    alpha = 0.2
    A = np.zeros((N, N))
    for i in range(N - 1):
        for j in range(i + 1, N):
            prob = expit(alpha - dst[i,j])
            A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)

    # np.savetxt('adj_simuC_3clusters.txt', A)
    # np.savetxt('label_simuC_3clusters.txt', Label)

    # To test LPCM package in R, we need to delete nodes not connected to others
    # a = np.sum(A, axis=0)
    # arr = np.delete(A, np.where(a == 0), axis=0)
    # arr = np.delete(arr, np.where(a == 0), axis=1)
    # lab = np.delete(Label, np.where(a == 0), axis=0)

    # np.savetxt('adj_simuC_3clusters_LPCM.txt', arr)
    # np.savetxt('label_simuC_3clusters_LPCM.txt', lab)

    return A, Label