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

# Train on CPU or GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')  # GPU
# print(torch.cuda.is_available())
# print(device)
# os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CPU
# device = 'cpu'

##################### Load data ########################
if args.dataset == 'cora' or args.dataset == 'simu1' or args.dataset == 'simu2' or args.dataset == 'simu3':
    features, adj, labels, model_embeddings = load_data(args.dataset)

    # # concatenate A with X
    # mat = np.concatenate((adj, features), axis=1)
    # mat = sp.csr_matrix(mat)

    adj = sp.csr_matrix(adj)
    bow = torch.from_numpy(features)  # used to calculate loss4
    # print(bow)
    features = sp.csr_matrix(features)
    embeddings = torch.from_numpy(model_embeddings).to(device)

##################### Some preprocessing ########################
adj_norm = preprocess_graph(adj)  # normalize adjacency matrix
# normalized_bow = preprocess_bow(features)  # normalize bow
# features = sparse_to_tuple(normalized_bow.tocoo())

# normalize bow par la norm en ligne
sums = torch.sqrt(torch.sum(bow ** 2, dim=1))
normalize_bow = bow / sums.unsqueeze(1)
# print(normalize_bow)
normalize_bow = sp.csr_matrix(normalize_bow)
features = sparse_to_tuple(normalize_bow.tocoo())

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
                {"params": model.decoder2.parameters(), 'lr': 0.02},
                {"params": model.alpha.parameters(), 'lr': 0.02},
                {"params": model.w2.parameters(), 'lr': 0.02},
            ]
optimizer = Adam(grouped_parameters, lr=args.learning_rate)  # model.parameters
# optimizer = Adam(model.parameters(), lr=args.learning_rate)

# store loss
store_loss = torch.zeros(args.num_epoch).to(device)  # total loss
store_loss1 = torch.zeros(args.num_epoch).to(device)  # graph loss
store_loss2 = torch.zeros(args.num_epoch).to(device)  # kl loss
store_loss3 = torch.zeros(args.num_epoch).to(device)  # cluster loss
store_loss4 = torch.zeros(args.num_epoch).to(device)  # text loss
store_ari = []

def ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred, B_pred, P):
    # Graph reconstruction loss
    OO = adj_label.to_dense()*(torch.log((A_pred/(1. - A_pred)) + 1e-16)) + torch.log((1. - A_pred) + 1e-16)
    OO = OO.fill_diagonal_(0)
    OO = OO.to(device)
    # loss1 = -torch.sum(OO) / args.num_points
    loss1 = -OO.sum(1).mean()  # N ** 2 !
    # loss1 = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))

    # KL divergence
    det = 1e-10
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
    # loss2 = (gamma * KL).sum(1) / args.num_points
    ts = gamma * KL
    # deal with nan value
    # print(torch.any(torch.isnan(gamma * KL)))
    # my_tensor_np = ts.cpu().numpy()
    # my_tensor_np[np.isnan(my_tensor_np)] = 0.0
    # ts.copy_(torch.from_numpy(my_tensor_np).cuda())
    # print(torch.any(torch.isnan(ts)))

    ts_sum = torch.sum(ts)
    # my_sum_np = ts_sum.cpu().numpy()
    # my_sum_np[np.isnan(my_sum_np)] = 0.0
    # ts_sum.copy_(torch.from_numpy(my_sum_np).cuda())

    loss2 = ts_sum / (args.num_points * args.num_clusters)
    # print('1:', ts)
    # print('2:', ts_sum)
    # print('3',loss2)

    # clustering loss
    # print(pi_k)
    loss3 = (gamma * (torch.log(pi_k.unsqueeze(0)) - torch.log(gamma))).sum(1).mean()

    # reconstruction text loss
    loss4 = -(B_pred * bow.to(device)).sum(1).mean()  # N * Q!

    loss = loss1 + loss2 - loss3 + loss4  # total loss

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

    if (epoch + 1) % 1 == 0:
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

end = time.time()
print('training time ......................:', end-begin)


################################# plots to show results ###################################
# plot train loss
f, ax = plt.subplots(1, figsize=(15, 10))
plt.subplot(231)
plt.plot(store_loss1.cpu().data.numpy(), color='red')
plt.title("Reconstruction graph loss")

plt.subplot(232)
plt.plot(store_loss2.cpu().data.numpy(), color='red')
plt.title("KL loss")

plt.subplot(233)
plt.plot(store_loss3.cpu().data.numpy(), color='red')
plt.title("Cluster loss")

plt.subplot(234)
plt.plot(store_loss4.cpu().data.numpy(), color='red')
plt.title("Recontruction text loss")

plt.subplot(235)
plt.plot(store_loss.cpu().data.numpy(), color='red')
plt.title("Training loss in total")

plt.subplot(236)
plt.plot(store_ari, color='blue')
plt.title("ARI")

plt.show()

print('Min loss:', torch.min(store_loss), 'K='+str(args.num_clusters))
print('ARI_gamma:', adjusted_rand_score(labels, torch.argmax(gamma, axis=1).cpu().numpy()))
print('Topics:', get_topics(beta.cpu().data.numpy()))
# print('Topic coherence:', get_topic_coherence(beta.cpu().data.numpy(), bow))

# ARI with kmeans
kmeans = KMeans(n_clusters=args.num_clusters).fit(eta.cpu().data.numpy())
labelk = kmeans.labels_
print("ARI_kmeans:", adjusted_rand_score(labels, labelk))

# visu
labelC = []
for idx in range(len(labels)):
    if labels[idx] == 0:
        labelC.append('lightblue')
    elif labels[idx] == 1:
        labelC.append('lightgreen')
    else:
        labelC.append('orange')

# visu of Z
pca = PCA(n_components=2, svd_solver='full')
out = pca.fit_transform(model.encoder.mean.cpu().data.numpy())
mean = pca.fit_transform(model.mu_k.cpu().data.numpy())
f, ax = plt.subplots(1, figsize=(15, 10))
ax.scatter(out[:, 0], out[:, 1], color=labelC)
ax.scatter(mean[:, 0], mean[:, 1], color='black', s=50)
ax.set_title('PCA result of Z of GETM (K='+str(args.num_clusters)+')', fontsize=18)
plt.show()

# visu of eta
pca = PCA(n_components=2, svd_solver='full')
out1 = pca.fit_transform(eta.cpu().data.numpy())
f, ax = plt.subplots(1, figsize=(15, 10))
ax.scatter(out1[:, 0], out1[:, 1], color=labelC)
ax.set_title('PCA result of $\eta$ of GETM (K='+str(args.num_clusters)+')', fontsize=18)
plt.show()

# visu of theta 2d
label_text = np.loadtxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_text_SBM_2.txt')
labelC_text = []
for idx in range(len(label_text)):
    if label_text[idx] == 0:
        labelC_text.append('pink')
    elif label_text[idx] == 1:
        labelC_text.append('purple')
    else:
        labelC_text.append('red')
# pca2 = PCA(n_components=2, svd_solver='full')
# out2 = pca2.fit_transform(theta.cpu().data.numpy())
out2 = theta.cpu().data.numpy()
f, ax = plt.subplots(1, figsize=(15, 10))
ax.scatter(out2[:, 0], out2[:, 1], color=labelC_text)
ax.set_title('PCA result of $\\theta$ of GETM (T='+str(args.num_topics)+')', fontsize=18)
plt.show()

# visu of theta 3d
from mpl_toolkits.mplot3d import Axes3D
# pca = PCA(n_components=3)
# out4 = pca.fit_transform(model.encoder.mean.cpu().data.numpy())
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=out2[:,0],
    ys=out2[:,1],
    zs=out2[:,2],
    color=labelC_text
)
ax.set_xlabel('topic-one')
ax.set_ylabel('topic-two')
ax.set_zlabel('topic-three')
plt.show()


########################### save data for visualisation in R ##################################
# import csv
# file = open('cora_data_A_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.csv', "w")
# writer = csv.writer(file)
# mean = model.encoder.mean.cpu().data.numpy()
# pred_labels = torch.argmax(gamma, axis=1).cpu().numpy()
# for w in range(args.num_points):
#     writer.writerow([w, mean[w][0],mean[w][1],mean[w][2],mean[w][3],mean[w][4],mean[w][5],mean[w][6],mean[w][7],
#                      mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15], pred_labels[w]])  # mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15]
# file.close()
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, svd_solver='full')
# out = pca.fit_transform(mean)
# np.savetxt('cora_pos_A_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', out)
#
# np.savetxt('cora_cl_A_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', pred_labels)
#
# np.savetxt('cora_mu_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', model.mu_k.cpu().data.numpy())