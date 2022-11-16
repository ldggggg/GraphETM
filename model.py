import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import itertools
import os
import numpy as np
import args
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
from input_data import load_data
from sklearn.cluster import KMeans
from preprocessing import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')
# os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CPU
# device = 'cpu'

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim, dtype = torch.float32) * 2 * init_range - init_range
    initial = initial.to(device)
    return nn.Parameter(initial)


# Graph convolutional layers
class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs  # node features matrix X or hidden output
        # print("2", x.shape)
        w = torch.mm(x, x.T)  # W2 dot product
        # x = torch.mm(x, self.weight)
        # x = torch.mm(self.adj, x)  # adj: N * N
        x = torch.mul(w, self.adj.to_dense())  # element-wise multiplication
        x = torch.mm(x, self.weight)
        outputs = self.activation(x)
        return outputs

class GraphConvNew(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvNew, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj  # mat
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        # print("1", x.shape)
        w = torch.mm(x, x.to_dense().T)  # W2
        # print(x)
        # sums = x.sum(1)
        # normalized_x = x / sums.unsqueeze(1)  # normalized x
        # print(normalized_x)
        # x = torch.mm(normalized_x, self.adj.to_dense())
        # print(x)
        x = torch.mul(w, self.adj.to_dense())  # *
        x = torch.mm(x, self.weight)
        # x1 = x.cpu().numpy()
        # x = preprocess_x(x1)  # normalize the input
        # print(x)
        # x = x.to(device)
        outputs = self.activation(x)
        return outputs


class Encoder(nn.Module):
    def __init__(self, adj_norm):
        super(Encoder, self).__init__()
        # self.base_gcn = GraphConvSparse(args.vocab_size, args.hidden1_dim, adj_norm)
        self.base_gcn = GraphConvNew(args.num_points, args.hidden1_dim, adj_norm)
        self.gcn_mean = GraphConvSparse(args.num_points, args.hidden2_dim, adj_norm, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.num_points, 1, adj_norm, activation=lambda x: x)
        # self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj_norm, activation=lambda x: x)
        # self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, 1, adj_norm, activation=lambda x: x)

        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        hidden = self.base_gcn(X)
        hidden = self.dropout(hidden)
        self.mean = self.gcn_mean(hidden)  # N * P
        self.logstd = self.gcn_logstddev(hidden)  # N * 1
        gaussian_noise = torch.randn(args.num_points, args.hidden2_dim)
        gaussian_noise = gaussian_noise.to(device)
        sampled_Z = gaussian_noise * torch.exp(self.logstd / 2) + self.mean  # embeddings
        return self.mean, self.logstd, sampled_Z


class Decoder_graph(nn.Module):
    # input: N * P.
    def __init__(self):
        super(Decoder_graph, self).__init__()

    def forward(self, eta, a):
        inner_product = torch.matmul(eta, eta.T)
        tnp = torch.sum(eta ** 2, dim=1).reshape(-1, 1).expand(size=inner_product.shape)
        A_pred = torch.sigmoid(- (tnp - 2 * inner_product + tnp.T) + a)
        # A_pred = torch.sigmoid(inner_product)
        return A_pred


class Decoder_text(nn.Module):
    def __init__(self):
        super(Decoder_text, self).__init__()

    # reconstruct bow
    def forward(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds


class GETM(nn.Module):
    def __init__(self, adj_norm, embeddings):
        super(GETM, self).__init__()
        self.adj_norm = adj_norm
        self.encoder = Encoder(adj_norm)
        self.decoder1 = Decoder_graph()
        self.decoder2 = Decoder_text()

        self.a = nn.Parameter(torch.tensor(0.2, dtype = torch.float32), requires_grad=True)  # graph decoder parameter

        # clustering parameters
        self.gamma = nn.Parameter(torch.FloatTensor(args.num_points, args.num_clusters).fill_(0.1),
                                  requires_grad=False)  # N * K
        self.pi_k = nn.Parameter(torch.FloatTensor(args.num_clusters, ).fill_(1) / args.num_clusters,
                                 requires_grad=False)  # K
        self.mu_k = nn.Parameter(torch.FloatTensor(args.num_clusters, args.hidden2_dim).fill_(0.1), requires_grad=False)  # K * P
        # self.mu_k = nn.Parameter(torch.FloatTensor(np.random.multivariate_normal(np.zeros(args.hidden2_dim),
        #                                                                          np.eye(args.hidden2_dim),
        #                                                                          args.num_clusters)),
        #                          requires_grad=False)
        self.log_cov_k = nn.Parameter(torch.FloatTensor(args.num_clusters, 1).fill_(0.1), requires_grad=False)  # K

        # topic parameters
        # self.rho = nn.Linear(args.rho_size, args.vocab_size, bias=False)  # L * V
        self.rho = embeddings.clone().float()  # word2vec embeddings
        self.alpha = nn.Linear(args.rho_size, args.num_topics, bias=False)  # L * K

        # define weights matrices for graph and topic embeddings
        self.w1 = nn.Linear(args.hidden2_dim, args.g_hidden_dim, bias=True)  # weights for graph embeddings
        self.w2 = nn.Linear(args.hidden2_dim, args.num_topics, bias=True)  # weights for topics

        # self.tau = nn.Parameter(torch.tensor(0.5, dtype = torch.float32), requires_grad=True)


    # pre-train of graph embeddings Z to initialize parameters of cluster
    def pretrain(self, X, adj_label, labels):
        if not os.path.exists('./pretrain_model.pk'):

            # when simu: weight_decay=1e-4;
            optimizer = Adam(itertools.chain(self.encoder.parameters(), self.decoder1.parameters()), lr=args.pre_lr, weight_decay=1e-4)  # , weight_decay=1e-4

            store_pre_loss = torch.zeros(args.pre_epoch)
            for epoch in range(args.pre_epoch):
                z_mu, z_log_sigma, z = self.encoder(X)
                A_pred = self.decoder1(z, self.a)
                loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))  # simply use reconstruction loss
                kl_divergence = 0.5 / A_pred.size(0) * (
                            1 + 2 * z_log_sigma - z_mu ** 2 - torch.exp(z_log_sigma) ** 2).sum(1).mean()
                loss -= kl_divergence

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 1 == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()))
                store_pre_loss[epoch] = torch.Tensor.item(loss)

            with torch.no_grad():
                Z = z.detach().cpu().numpy()

            # plot loss
            # f, ax = plt.subplots(1, figsize=(15, 10))
            # ax.plot(store_pre_loss, color='red')
            # ax.set_title("pre-training loss")
            # plt.show()

            # Additional lines to initialize gamma based on the k-means on the latent embeddings
            kmeans = KMeans(n_clusters=args.num_clusters).fit(Z)
            labelk = kmeans.labels_
            print("ARI_kmeans:", adjusted_rand_score(labels, labelk))

            self.gamma.fill_(1e-10)
            seq = np.arange(0, len(self.gamma))
            positions = np.vstack((seq, labelk))
            self.gamma[positions] = 1.
            # print(self.gamma)

            self.mu_k.data = torch.from_numpy(kmeans.cluster_centers_).float().to(device)  # need in simu1_pi=0.8, simu2 !

            ################## gmm #################
            # gmm = GaussianMixture(n_components=args.num_clusters, covariance_type='diag')
            # pre = gmm.fit_predict(Z)
            # print("pretraining ARI GMM:", adjusted_rand_score(labels, pre))
            # # pre = np.argmax(np.random.multinomial(1, [1 / args.num_clusters] * args.num_clusters, size=args.num_points),
            # #                 axis=1)
            # # print(pre)
            # # print("pretraining ARI random:", adjusted_rand_score(labels, pre))
            #
            # # self.pi_k.data = torch.from_numpy(gmm.weights_).float().to(device)
            # # self.mu_k.data = torch.from_numpy(gmm.means_).float().to(device)
            # # self.log_cov_k.data = torch.log(torch.from_numpy(gmm.covariances_)).float().to(device)
            # self.gamma.fill_(1e-10)
            # seq = np.arange(0, len(self.gamma))
            # positions = np.vstack((seq, pre))
            # self.gamma[positions] = 1.
            # print(self.gamma)

            # visu
            # labelC = []
            # for idx in range(len(labels)):
            #     if labelk[idx] == 0:
            #         labelC.append('lightblue')
            #     elif labelk[idx] == 1:
            #         labelC.append('lightgreen')
            #     elif labelk[idx] == 2:
            #         labelC.append('yellow')
            #     elif labelk[idx] == 3:
            #         labelC.append('pink')
            #     elif labelk[idx] == 4:
            #         labelC.append('purple')
            #     elif labelk[idx] == 5:
            #         labelC.append('red')
            #     else:
            #         labelC.append('orange')
            # f, ax = plt.subplots(1, figsize=(15, 10))
            # ax.scatter(Z[:, 0], Z[:, 1], color=labelC)
            # plt.show()

            # torch.save(self.state_dict(), './pretrain_model.pk')
            print('Finish pretraining!')

        else:
            print('Loading...............')
            self.load_state_dict(torch.load('./pretrain_model.pk'))
            # print('pi:', self.pi_k)
            # print('mu:', self.mu_k)
            # print('cov:', self.log_cov_k)

    # Functions for the initialization of cluster parameters
    def update_gamma(self, mu_phi, log_cov_phi, pi_k, mu_k, log_cov_k, P):
        det = 1e-16
        KL = torch.zeros((args.num_points, args.num_clusters), dtype = torch.float32)  # N * K
        KL = KL.to(device)
        for k in range(args.num_clusters):
            log_cov_K = torch.ones_like(log_cov_phi) * log_cov_k[k]
            mu_K = torch.ones((args.num_points, mu_k.shape[1])).to(device) * mu_k[k]
            temp = P * (log_cov_K - log_cov_phi - 1) \
                   + P * torch.exp(log_cov_phi) / torch.exp(log_cov_K) \
                   + torch.norm(mu_K - mu_phi, dim=1, keepdim=True) ** 2 / torch.exp(log_cov_K)
            KL[:, k] = 0.5 * temp.squeeze()
        # print(KL)
        denominator = torch.sum(pi_k.unsqueeze(0) * torch.exp(-KL), dim=1, dtype = torch.float32) + det
        for k in range(args.num_clusters):
            self.gamma.data[:, k] = pi_k[k] * torch.exp(-KL[:, k]) / denominator + det

    def update_others(self, mu_phi, log_cov_phi, gamma, P):
        N_k = torch.sum(gamma, dim=0, dtype = torch.float32)
        # print('gamma:', gamma)
        # print('N_k:', N_k)
        self.pi_k.data = N_k / args.num_points

        for k in range(args.num_clusters):
            gamma_k = gamma[:, k]  # N * 1
            self.mu_k.data[k] = torch.sum(mu_phi * gamma_k.unsqueeze(1), dim=0, dtype = torch.float32) / N_k[k]
            mu_k = self.mu_k

            diff = P * torch.exp(log_cov_phi) + torch.sum((mu_k[k].unsqueeze(0) - mu_phi) ** 2, dim=1, dtype = torch.float32).unsqueeze(1)
            cov_k = torch.sum(gamma_k.unsqueeze(1) * diff, dim=0, dtype = torch.float32) / (P * N_k[k])
            self.log_cov_k.data[k] = torch.log(cov_k)

    def get_beta(self):
        # try:
        #     # torch.mm(self.rho, self.alphas)
        #     logit = self.alpha(self.rho.weight)
        # except BaseException:
        logit = self.alpha(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0)  # softmax over vocab dimension
        return beta

    #  get graph embeddings
    def get_eta(self, z):
        # print(z[0])
        eta = self.w1(z)
        # eta = z  # TODO!!! Do we need this w1?
        # print(eta[0])
        return eta

    #  get topic proportions
    def get_theta(self, z):
        # sums = z.sum(1)
        # print('z:',z)
        # normalized_z = z / sums.unsqueeze(1)  # normalized as ETM
        # print('norm_z:',normalized_z)
        delta = self.w2(z)
        theta = F.softmax(delta, dim=-1)
        return delta, theta