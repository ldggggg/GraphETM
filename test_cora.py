# # Load data
# path = 'C:/Users/Dingge/Documents/GitHub/etm-deeplpm/etm-deeplpm/data/Cora_enrich/'
# with open(path + "idxs.txt", "r") as f:
#     idxs = f.readlines()
#     idxs = [int(x) for x in idxs]
#
# # change index to [0,2707]
# a = []
# for i in idxs:
#     a.append(idxs.index(i))
# b = idxs
# c = zip(b,a)
# dic = dict(c)
#
# with open(path + "labels.txt", "r") as f:
#     labels_raw = f.readlines()
#     labels_raw = [x.strip("\n") for x in labels_raw]
#
# map_label_str2int = {lbl: i for i, lbl in enumerate(list(set(labels_raw)))}
# labels_int = [map_label_str2int[x] for x in labels_raw]
#
#
# with open(path + "links.txt", "r") as f:
#     links = f.readlines()
#     # links = [x.split() for x in links]
#     links = [list(map(int, x.split())) for x in links]
#
# # create adj
# import numpy as np
#
# num = len(idxs)
# matrix = np.zeros((num,num))
#
# cites = links
# for j in cites:
#     if len(j) > 0:
#         x = cites.index(j)
#         for m in j:
#             y = j.index(m)
#             cites[x][y] = dic[m]  # index to [0,2707]
#
# for i, j in zip(idxs, links):
#     x = dic[i]  # index to [0,2707]
#     for y in j:
#         matrix[x][y] = matrix[y][x] = 1
# print(sum(matrix))
#
# np.savetxt(path + "cora_adj.txt", matrix)
# np.savetxt(path + "cora_labels_int.txt", labels_int)


import preprocessing

path = 'C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/'
# path = '/home/dliang/etm-deeplpm/etm_deeplpm/data/Cora_enrich/'
with open(path + "texts.txt", "r") as f:
    txts = f.readlines()
    txts = [x.strip("\n") for x in txts]

documents = txts
doc = [x.split(' ') for x in documents]


# Preprocessing the dataset
vocabulary, train_dataset, _, = preprocessing.create_etm_datasets(
    documents,
    min_df=0.01,  # 0.01
    max_df=0.75,  # 0.75
    train_size=1  # 0.85
)

import pickle
with open('C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/cora_docs', 'wb') as fp:
    pickle.dump(vocabulary, fp)

# from etm_deeplpm.utils import embedding
#
# # Training word2vec embeddings
# embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(documents)
# #
# # Save model
# with open(path + '/tmp/word2vec_model_ori.pkl', 'wb') as fout:
#     pickle.dump((embeddings_mapping), fout)

# Load savec model
with open(path + 'word2vec_model_1.pkl', 'rb') as f:
    embeddings_mapping_save = pickle.load(f)

from etm_dlpm import ETM_DLPM
import torch

# Training ETM with DeepLPM
etm_instance = ETM_DLPM(
    vocabulary,
    embeddings=embeddings_mapping_save,  # You can pass here the path to a word2vec file or
                                   # a KeyedVectors instance
    z_hidden_size=16,
    num_topics=20,
    num_clusters=7,
    g_hidden2_size=16,
    epochs=20,
    lr=0.0001,
    debug_mode=True,
    train_embeddings=False, # Optional. If True, ETM will learn word embeddings jointly with
                            # topic embeddings. By default, is False. If 'embeddings' argument
                            # is being passed, this argument must not be True
)
# load pre-training ETM
# etm_instance.model.load_state_dict(torch.load('./pretrain_model_1.pk'))

save_loss = etm_instance.fit(train_dataset)

topics = etm_instance.get_topics(10)
topic_coherence = etm_instance.get_topic_coherence()
topic_diversity = etm_instance.get_topic_diversity()


import matplotlib.pyplot as plt
# plot train loss
f, ax = plt.subplots(1, figsize=(15, 10))
plt.subplot(241)
plt.plot(save_loss[1], color='red')
plt.title("Reconstruction graph loss")

plt.subplot(242)
plt.plot(save_loss[2], color='red')
plt.title("Reconstruction text loss")

plt.subplot(243)
plt.plot(save_loss[3], color='red')
plt.title("KL divergence")

plt.subplot(244)
plt.plot(save_loss[0], color='red')
plt.title("Training loss in total")

plt.subplot(245)
plt.plot(save_loss[4], color='red')
plt.title("Cluster loss")

plt.subplot(246)
plt.plot(save_loss[5], color='green')
plt.title("Training accuracy")

plt.subplot(247)
plt.plot(save_loss[6], color='blue')
plt.title("Training ari gamma")

plt.subplot(248)
plt.plot(save_loss[7], color='blue')
plt.title("Training ari kmeans")

f.savefig(path + "/tmp/loss_acc_ari_new.pdf", bbox_inches='tight')


######################################## SBM ##############################################
from input_data import *
from sklearn.metrics.cluster import adjusted_rand_score

# features, adj, labels, model_embeddings = load_data('simu1')
adj = np.loadtxt("data/SBM/adj_SBM_1_pi=0.6.txt")
labels = np.loadtxt("data/SBM/label_SBM_1.txt")

from sparsebm import SBM
number_of_clusters = 3
# A number of classes must be specify. Otherwise see model selection.
model1 = SBM(number_of_clusters, n_init=100)
model1.fit(adj, symmetric=True)
# print("Labels:", model.labels)
print("ARI_SBM:", adjusted_rand_score(labels, model1.labels))

# from SBM_package.src import SBM
# elbo, tau, tau_init, count, time_list = SBM.sbm(adj, 3, algo='vbem', type_init='kmeans')
# c = np.argmax(tau, axis=1)
# print("ARI_SBM_init_kmeans:", adjusted_rand_score(labels, c))


######################### plot simplex #######################
import plotly.express as px
import matplotlib.pyplot as plt

import plotly.io as pio
pio.renderers
pio.renderers.default = "browser"

df = px.data.election()
fig = px.scatter_ternary(df, a="Joly", b="Coderre", c="Bergeron")
fig.show()