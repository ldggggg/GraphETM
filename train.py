import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt

from input_data import *
from preprocessing import *
import model
import args
from evaluation import eva

# Train on CPU or GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')  # GPU
# print(torch.cuda.is_available())
# print(device)
# os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CPU
# device = 'cpu'

##################### Load data ########################
if args.dataset == 'simu1' or args.dataset == 'simu2' or args.dataset == 'simu3':
    features, adj, labels, model_embeddings, label_text, dct = load_data(args.dataset)
elif args.dataset == 'cora':
    features, adj, labels, model_embeddings, label_text, dct = load_data(args.dataset)

# # concatenate A with X
# mat = np.concatenate((adj, features), axis=1)
# mat = sp.csr_matrix(mat)

adj = sp.csr_matrix(adj)
bow = torch.from_numpy(features)  # used to calculate loss4
# print(bow)
features = sp.csr_matrix(features)  # bow
embeddings = torch.from_numpy(model_embeddings).to(device)

# ############### SBM #################
# from sparsebm import SBM
# number_of_clusters = args.num_clusters
# # A number of classes must be specify. Otherwise see model selection.
# model1 = SBM(number_of_clusters, n_init=100)
# model1.fit(adj.todense(), symmetric=True)
# # print("Labels:", model.labels)
# ari_sbm = adjusted_rand_score(labels, model1.labels)

# from SBM_package.src import SBM
# elbo, tau, tau_init, count, time_list = SBM.sbm(adj.todense(), args.num_clusters, algo='vbem', type_init='kmeans')
# c = np.argmax(tau, axis=1)
# print("ARI_SBM_init_kmeans:", adjusted_rand_score(labels, c))

##################### Some preprocessing ########################
adj_norm = preprocess_graph(adj)  # normalize adjacency matrix
# normalized_bow = preprocess_bow(features)  # normalize bow
# features = sparse_to_tuple(normalized_bow.tocoo())

# normalize bow par la norm en ligne
sums = torch.sqrt(torch.sum(bow ** 2, dim=1))
normalize_bow = bow / sums.unsqueeze(1)
# print(normalize_bow)
normalize_bow = sp.csr_matrix(normalize_bow)
features = sparse_to_tuple(normalize_bow.tocoo())  # normalized bow


# # [A . WW^T]
# features = sparse_to_tuple(features.tocoo())
# W = np.dot(normalize_bow, normalize_bow.T)  # WW^T
# mul = np.multiply(adj_norm.todense(), W)
# adj_norm = sparse_to_tuple(sp.csr_matrix(mul).tocoo())

#features = sparse_to_tuple(features.tocoo())
adj_ori = sparse_to_tuple(adj)  # original adj
# mat = sparse_to_tuple(mat.tocoo())
# mat = preprocess_mat(mat)  # normalize concatnated matrix
adj_label = adj + sp.eye(adj.shape[0])  # used to calculate the loss
adj_label = sparse_to_tuple(adj_label)

# Create Model
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].astype(float).T),
                            torch.FloatTensor(adj_norm[1].astype(float)),
                            torch.Size(adj_norm[2]))
adj_ori = torch.sparse.FloatTensor(torch.LongTensor(adj_ori[0].astype(float).T),
                            torch.FloatTensor(adj_ori[1].astype(float)),
                            torch.Size(adj_ori[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].astype(float).T),
                            torch.FloatTensor(adj_label[1]),
                            torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].astype(float).T),
                            torch.FloatTensor(features[1]), 
                            torch.Size(features[2]))
# mat = torch.sparse.FloatTensor(torch.LongTensor(mat[0].astype(float).T),
#                             torch.FloatTensor(mat[1]),
#                             torch.Size(mat[2]))

# to GPU
adj_norm = adj_norm.to(device)
adj_label = adj_label.to(device)
features = features.to(device)
adj_ori = adj_ori.to(device)

################################ Model ##################################
# init model and optimizer
model = getattr(model, args.model)(adj_norm, embeddings)
model.to(device)  # to GPU

model.pretrain(features, adj_label, labels)  # pretraining

# set different lr for two parts
grouped_parameters = [
                {"params": model.encoder.parameters(), 'lr': 5e-3},
                {"params": model.a, 'lr': 5e-3},
                {"params": model.decoder1.parameters(), 'lr': 5e-3},
                {"params": model.w1.parameters(), 'lr': 5e-3},
                {"params": model.decoder2.parameters(), 'lr': 0.01},
                {"params": model.alpha.parameters(), 'lr': 0.01},
                {"params": model.w2.parameters(), 'lr': 0.01},
                # {"params": model.tau, 'lr': 5e-3},
            ]  # simu: 5e-3, 0.02; cora:5e-3, 0.01.
optimizer = Adam(grouped_parameters, lr=args.learning_rate)  # model.parameters, weight_decay=1e-4
# optimizer = Adam(model.parameters(), lr=args.learning_rate)
lr_s = StepLR(optimizer, step_size=50, gamma=0.1)  # 100 for cora, 50 when g_dim=128. 100 for simu.

# store loss
store_loss = torch.zeros(args.num_epoch).to(device)  # total loss
store_loss1 = torch.zeros(args.num_epoch).to(device)  # graph loss
store_loss2 = torch.zeros(args.num_epoch).to(device)  # kl loss
store_loss3 = torch.zeros(args.num_epoch).to(device)  # cluster loss
store_loss4 = torch.zeros(args.num_epoch).to(device)  # text loss
store_ari = []
store_acc = []

def ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred, B_pred, P):
    # Graph reconstruction loss
    OO = adj_label.to_dense()*(torch.log((A_pred/(1. - A_pred)) + 1e-16)) + torch.log((1. - A_pred) + 1e-16)
    OO = OO.fill_diagonal_(0)
    OO = OO.to(device)
    # loss1 = -torch.sum(OO) / (args.num_points * args.num_points)
    # print(OO.shape)
    loss1 = -OO.sum(1).mean()  # N ** 2 !
    # loss1 = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))

    # KL divergence
    det = 1e-16
    KL = torch.zeros((args.num_points, args.num_clusters))  # N * K
    KL = KL.to(device)
    for k in range(args.num_clusters):
        log_cov_K = torch.ones_like(log_cov_phi) * log_cov_k[k]
        mu_K = torch.ones((args.num_points, mu_k.shape[1])).to(device) * mu_k[k]
        temp = P*(log_cov_K-log_cov_phi-1) \
                  + P*torch.exp(log_cov_phi)/torch.exp(log_cov_K) \
                  + torch.norm(mu_K-mu_phi,dim=1,keepdim=True)**2/torch.exp(log_cov_K)
        KL[:, k] = 0.5*temp.squeeze()

    # kl loss
    loss2 = (gamma * KL).sum(1).mean()
    # ts = gamma * KL
    # ts_sum = torch.sum(ts)
    # loss2 = ts_sum / (args.num_points * args.num_clusters)

    # clustering loss
    # print(pi_k)
    loss3 = (gamma * (torch.log(pi_k.unsqueeze(0)) - torch.log(gamma))).sum(1).mean()
    # loss3 = torch.sum(gamma * (torch.log(pi_k.unsqueeze(0)) - torch.log(gamma))) / (args.num_points * args.num_clusters)

    # reconstruction text loss
    # print(B_pred.shape)
    loss4 = -(B_pred * bow.to(device)).sum(1).mean()  # N * Q!


    loss = loss1 + loss2 - loss3 + loss4  # total loss, cora: loss4 * 0.01

    return loss, loss1, loss2, -loss3, loss4


##################### Visualisation of learned embeddings by PCA ####################
from sklearn.decomposition import PCA
def visu():
    if args.dataset == 'simu1':
        labelC = []
        for idx in range(len(labels)):
            if labels[idx] == 0:
                labelC.append('lightblue')
            elif labels[idx] == 1:
                labelC.append('lightgreen')
            else:
                labelC.append('yellow')
        pca = PCA(n_components=2, svd_solver='full')
        out = pca.fit_transform(model.encoder.mean.cpu().data.numpy())
        mean = pca.fit_transform(model.mu_k.cpu().data.numpy())
        f, ax = plt.subplots(1, figsize=(15, 10))
        ax.scatter(out[:, 0], out[:, 1], c=labelC)
        ax.scatter(mean[:, 0], mean[:, 1], color='black', s=50)
        ax.set_title('PCA result of embeddings Z of GETM (K='+str(args.num_clusters)+')', fontsize=18)
        plt.show()
        # f.savefig("C:/Users/Dingge/Desktop/results/emb_ARVGA.pdf", bbox_inches='tight')

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

#################################### train model ################################################
begin = time.time()
for epoch in range(args.num_epoch):
    t = time.time()

    # get latent embeddings
    mu_phi, log_cov_phi, z = model.encoder(features)  # features

    # reconstruct graph
    eta = model.get_eta(z)
    A_pred = model.decoder1(eta, model.a)

    # reconstruct bow
    beta = model.get_beta()
    delta, theta = model.get_theta(z)
    B_pred = model.decoder2(theta, beta)

    with torch.no_grad():
        # if epoch < 1 or (epoch + 1) % 1 == 0:
        ################# update pi_k, mu_k and log_cov_k ###################
        # print('1:',model.pi_k, model.gamma)
        gamma = model.gamma.data
        model.update_others(mu_phi.detach().clone(),
                            log_cov_phi.detach().clone(),
                            gamma, args.hidden2_dim)

        # update gamma
        pi_k = model.pi_k.data
        # print('2',pi_k, model.gamma)
        log_cov_k = model.log_cov_k.data
        mu_k = model.mu_k.data
        model.update_gamma(mu_phi.detach().clone(),
                           log_cov_phi.detach().clone(),
                           pi_k, mu_k, log_cov_k, args.hidden2_dim)

    # with torch.no_grad():
        pi_k = model.pi_k.data                    # pi_k should be a copy of model.pi_k
        # print('3',pi_k, model.gamma)
        log_cov_k = model.log_cov_k.data
        mu_k = model.mu_k.data
        gamma = model.gamma.data

    # calculate of ELBO loss
    optimizer.zero_grad()
    loss, loss1, loss2, loss3, loss4 = ELBO_Loss(gamma, pi_k, mu_k, log_cov_k,
                                                 mu_phi.detach().clone(), log_cov_phi.detach().clone(),
                                                 A_pred, B_pred, args.hidden2_dim)

    # if epoch > 1:
    ################ update of GCN ############
    loss.backward()
    optimizer.step()
    if epoch < 100:  # 100 when g_dim=128
        lr_s.step()  # need in simu3.

    if (epoch + 1) % 1 == 0:
        eva(labels, torch.argmax(gamma, axis=1).cpu().numpy(), epoch)
        print("Epoch:", '%04d' % (epoch + 1), "total_loss=", "{:.5f}".format(loss.item()),
              "reconstruct_graph_loss=", "{:.5f}".format(loss1.item()), "kl_loss=", "{:.5f}".format(loss2.item()),
              "cluster_loss=", "{:.5f}".format(loss3.item()), "reconstruct_text_loss=", "{:.5f}".format(loss4.item()),
              "time=", "{:.5f}".format(time.time() - t))

    # if (epoch + 1) % 100 == 0:
    #     visu()
        # f, ax = plt.subplots(1, figsize=(10, 15))
        # ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
        # ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
        # ax.set_title("Embeddings after training!")
        # plt.show()

    store_loss[epoch] = torch.Tensor.item(loss)  # save train loss for visu
    store_loss1[epoch] = torch.Tensor.item(loss1)
    store_loss2[epoch] = torch.Tensor.item(loss2)
    store_loss3[epoch] = torch.Tensor.item(loss3)
    store_loss4[epoch] = torch.Tensor.item(loss4)

    # store_ari[epoch] = torch.tensor(adjusted_rand_index(labels, torch.argmax(gamma, axis=1)))  # save ARI
    if args.dataset == 'eveques':
        print('Unsupervised data without true labels (no ARI) !')
    else:
        store_ari.append(adjusted_rand_score(labels, torch.argmax(gamma, axis=1).cpu().numpy()))
        store_acc.append(get_acc(A_pred, adj_label).cpu().data.numpy())

end = time.time()
print('training time ......................:', end-begin)

################################# plots to show results ###################################
# plot train loss
f, ax = plt.subplots(1, figsize=(15, 10))
plt.subplot(221)
plt.plot(store_loss1.cpu().data.numpy(), color='red')
plt.title("Reconstruction graph loss")

# plt.subplot(242)
# plt.plot(store_loss2.cpu().data.numpy(), color='red')
# plt.title("KL loss")
#
# plt.subplot(243)
# plt.plot(store_loss3.cpu().data.numpy(), color='red')
# plt.title("Cluster loss")

plt.subplot(222)
plt.plot(store_loss4.cpu().data.numpy(), color='red')
plt.title("Recontruction text loss")

plt.subplot(223)
plt.plot(store_loss.cpu().data.numpy(), color='red')
plt.title("Training loss in total")

plt.subplot(224)
plt.plot(store_ari, color='blue')
plt.title("ARI for clustering")

# plt.subplot(247)
# plt.plot(store_acc, color='blue')
# plt.title("ACC")

plt.show()
# f.savefig("data/loss_sc.c.pdf", bbox_inches='tight')

print('Min loss:', torch.min(store_loss), 'K='+str(args.num_clusters), 'T='+str(args.num_topics))
# print('Max ACC:', max(store_acc))
print('ARI_gamma:', adjusted_rand_score(labels, torch.argmax(gamma, axis=1).cpu().numpy()))
# print('Topics:', get_topics(beta.cpu().data.numpy(), dct))
# print('Topic coherence:', get_topic_coherence(beta.cpu().data.numpy(), bow))
# print("ARI_SBM:", ari_sbm)
# ARI with kmeans
kmeans = KMeans(n_clusters=args.num_clusters).fit(model.encoder.mean.cpu().data.numpy())
labelk = kmeans.labels_
print("ARI_kmeans:", adjusted_rand_score(labels, labelk))

# visu
labelC = []
gamma = model.gamma.cpu().data.numpy()
pred_labels = np.argmax(gamma, axis=1)
for idx in range(len(pred_labels)):
    if pred_labels[idx] == 0:
        labelC.append('lightblue')
    elif pred_labels[idx] == 1:
        labelC.append('lightgreen')
    elif pred_labels[idx] == 2:
        labelC.append('orange')
    elif pred_labels[idx] == 3:
        labelC.append('pink')
    elif pred_labels[idx] == 4:
        labelC.append('purple')
    elif pred_labels[idx] == 5:
        labelC.append('red')
    else:
        labelC.append('yellow')

# # visu of Z
# pca = PCA(n_components=2, svd_solver='full')
# pca.fit(model.encoder.mean.cpu().data.numpy())
# out_Z = pca.transform(model.encoder.mean.cpu().data.numpy())
# out_mu = pca.transform(model.mu_k.cpu().data.numpy())
# f, ax = plt.subplots(1, figsize=(15, 10))
# ax.scatter(out_Z[:, 0], out_Z[:, 1], color=labelC)
# ax.scatter(out_mu[:, 0], out_mu[:, 1], color='black', s=50)
# ax.set_title('PCA result of Z of GETM (K='+str(args.num_clusters)+')', fontsize=18)
# plt.show()
# # f.savefig("data/emb_Z_sc.c.pdf", bbox_inches='tight')
#
# # visu of eta
# pca = PCA(n_components=2, svd_solver='full')
# out_eta = pca.fit_transform(eta.cpu().data.numpy())
# # out_eta = pca.transform(eta.cpu().data.numpy())
# f, ax = plt.subplots(1, figsize=(15, 10))
# ax.scatter(out_eta[:, 0], out_eta[:, 1], color=labelC)
# ax.set_title('PCA result of $\eta$ of GETM (K='+str(args.num_clusters)+')', fontsize=18)
# plt.show()
# # f.savefig("data/emb_eta_sc.c.pdf", bbox_inches='tight')

# # visu of theta 2d
# label_text = np.loadtxt('data/SBM/label_text_SBM_1.txt')
# labelC_text = []
# for idx in range(len(label_text)):
#     if label_text[idx] == 0:
#         labelC_text.append('pink')
#     elif label_text[idx] == 1:
#         labelC_text.append('purple')
#     else:
#         labelC_text.append('red')
# pca2 = PCA(n_components=2, svd_solver='full')
# out2 = pca2.fit_transform(theta.cpu().data.numpy())
# # out2 = theta.cpu().data.numpy()
# f, ax = plt.subplots(1, figsize=(15, 10))
# ax.scatter(out2[:, 0], out2[:, 1], color=labelC)
# ax.set_title('visualisation of $\\theta$ of GETM (T='+str(args.num_topics)+')', fontsize=18)
# plt.show()

# the histogram of the data
# n, bins, patches = plt.hist(out2[:, 1], 50, density=True, facecolor='g', alpha=0.75)
# # plt.xlabel('Topics')
# # plt.ylabel('Probability')
# plt.title('visualisation of $\\theta$ of GETM (T='+str(args.num_topics)+')', fontsize=18)
# # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# # plt.xlim(40, 160)
# # plt.ylim(0, 0.03)
# plt.grid(True)
# plt.show()
# # f.savefig("data/hist_theta_sc.c.pdf", bbox_inches='tight')

#### simplex ####
# import pandas as pd
# df = pd.DataFrame(out2, columns = ['Topic1','Topic2','Topic3'])
#
# import plotly.express as px
# import matplotlib.pyplot as plt
# import plotly.io as pio
# pio.renderers
# pio.renderers.default = "browser"
# fig = px.scatter_ternary(df, a="Topic1", b="Topic2", c="Topic3")
# fig.show()

# visu of theta 3d
# from mpl_toolkits.mplot3d import Axes3D
# pca = PCA(n_components=3)
# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=out2[:,0],
#     ys=out2[:,1],
#     zs=out2[:,2],
#     color=labelC_text
# )
# ax.set_xlabel('topic-one')
# ax.set_ylabel('topic-two')
# ax.set_zlabel('topic-three')
# plt.show()


########################### save data for visualisation in R ##################################
with torch.no_grad():
    mu, log_cov, z = model.encoder(features)
    mean = z.cpu().data.numpy()
    # mean = model.encoder.mean.cpu().data.numpy()
    gamma = model.gamma.cpu().data.numpy()
    pred_labels = np.argmax(gamma, axis=1)
    eta = model.get_eta(z).cpu().data.numpy()
    delta, theta = model.get_theta(z)
    delta = delta.cpu().data.numpy()
    theta = theta.cpu().data.numpy()
    beta = model.get_beta().cpu().data.numpy()

#################### export to csv ######################
# import pandas as pd
# df = pd.DataFrame(np.hstack([mean, pred_labels.reshape((-1,1))]))
# df.to_csv('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_data_Z_d=128_k='+str(args.num_clusters)+'.csv', header=False)
#
# df2 = pd.DataFrame(np.hstack([eta, pred_labels.reshape((-1,1))]))
# df2.to_csv('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_data_eta_d=128_k='+str(args.num_clusters)+'.csv', header=False)
#
# df3 = pd.DataFrame(np.hstack([delta, pred_labels.reshape((-1,1))]))
# df3.to_csv('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_data_delta_d=7_k='+str(args.num_clusters)+'.csv', header=False)

# import csv
# file = open('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_data_Z_k='+str(args.num_clusters)+'.csv', "w")
# writer = csv.writer(file)
# for w in range(args.num_points):
#     writer.writerow([w, mean[w][0],mean[w][1],mean[w][2],mean[w][3],mean[w][4],mean[w][5],mean[w][6],mean[w][7],
#                      mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15], pred_labels[w]])  # mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15]
# file.close()

# file2 = open('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_data_eta_k='+str(args.num_clusters)+'.csv', "w")
# writer2 = csv.writer(file2)
# for w in range(args.num_points):
#     writer2.writerow([w, eta[w][0],eta[w][1],eta[w][2],eta[w][3],eta[w][4],eta[w][5],eta[w][6],eta[w][7],
#                      eta[w][8],eta[w][9],eta[w][10],eta[w][11],eta[w][12],eta[w][13],eta[w][14],eta[w][15], pred_labels[w]])
# file2.close()
#
# file3 = open('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_data_delta_k='+str(args.num_clusters)+'.csv', "w")
# writer3 = csv.writer(file3)
# for w in range(args.num_points):
#     writer3.writerow([w, delta[w][0],delta[w][1],delta[w][2],delta[w][3],delta[w][4],delta[w][5],delta[w][6], pred_labels[w]])
# file3.close()

# np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_pos_Z_k='+str(args.num_clusters)+'_p=128.txt', out_Z)
# np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_cl_A_k='+str(args.num_clusters)+'_p=128.txt', pred_labels)
# np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_mu_k='+str(args.num_clusters)+'_p=128.txt', out_mu)

############################### confusion matrix ####################################
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# labels = np.loadtxt('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_labels_int.txt').astype(int)
# pred_labels = np.loadtxt('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_cl_A_k=7_p=16.txt').astype(int)
cf_m = confusion_matrix(pred_labels, labels)

# import seaborn as sns
# fig = plt.figure()
# sns.heatmap(cf_m, annot=True, fmt='', cmap='Blues', xticklabels=['O1','O2','O3','O4','O5','O6','O7'],
#             yticklabels=['N1','N2','N3','N4','N5','N6','N7'])
# plt.ylabel('Cluster partition with covariate Y')
# plt.xlabel('Cluster partition without covariate Y')
# fig.savefig("C:/Users/Dingge/Desktop/results/cf_m.pdf", bbox_inches='tight')
#
disp = ConfusionMatrixDisplay(confusion_matrix=cf_m, display_labels=[0,1,2,3,4,5,6])
disp.plot()