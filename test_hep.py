
# def load_hep():
# Load node data
path = 'C:/Users/Dingge/Downloads/BiTe-GCN-main/data/word_data/hep-small/'
with open(path + "nodes.txt", "r") as f:
    idxs = f.readlines()
    idxs = [int(x) for x in idxs]

# change index to [0,396]
a = []
for i in idxs:
    a.append(idxs.index(i))
b = idxs
c = zip(b,a)
dic = dict(c)

with open(path + "labels.txt", "r") as f:
    labels_raw = f.readlines()
    labels_raw = [x.strip("\n") for x in labels_raw]

map_label_str2int = {lbl: i for i, lbl in enumerate(list(set(labels_raw)))}
labels_int = [map_label_str2int[x] for x in labels_raw]

# load links
with open(path + "links.txt", "r") as f:
    links = f.readlines()
    # links = [x.split() for x in links]
    links = [list(map(int, x.split())) for x in links]

# create adj
import numpy as np

num = len(idxs)
matrix = np.zeros((num,num))

cites = links
for j in cites:
    if len(j) > 0:
        x = cites.index(j)
        for m in j:
            y = j.index(m)
            if m in dic.keys():
                cites[x][y] = dic[m]  # index to [0,396]
            else:
                print(m)

for i, j in zip(idxs, links):
    # print(i, j)
    x = dic[i]  # index to [0,396]
    for y in j:
        if y in dic.keys():
            matrix[x][y] = matrix[y][x] = 1
print(sum(matrix))

# load text data
with open(path + "raw_text.txt", "r") as f:
    txts = f.readlines()
    txts = [x.strip("\n") for x in txts]

# removing stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
txts = [w for w in txts if not w in stop_words]

documents = txts
doc = [x.split(' ') for x in documents]

import embedding
import pickle

# Training word2vec embeddings
embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(documents, dim_rho=250)

# Save model
with open(path + 'word2vec_model_hep.pkl', 'wb') as fout:
    pickle.dump((embeddings_mapping), fout)

# # Load savec model
# with open(path + 'word2vec_model_hep.pkl', 'rb') as f:
#     embeddings_mapping_save = pickle.load(f)


# from etm_dlpm import ETM_DLPM
# import torch
#
# # Training ETM with DeepLPM
# etm_instance = ETM_DLPM(
#     vocabulary,
#     embeddings=embeddings_mapping_save,  # You can pass here the path to a word2vec file or
#                                    # a KeyedVectors instance
#     z_hidden_size=16,
#     num_topics=20,
#     num_clusters=7,
#     g_hidden2_size=16,
#     epochs=20,
#     lr=0.0001,
#     debug_mode=True,
#     train_embeddings=False, # Optional. If True, ETM will learn word embeddings jointly with
#                             # topic embeddings. By default, is False. If 'embeddings' argument
#                             # is being passed, this argument must not be True
# )
# # load pre-training ETM
# # etm_instance.model.load_state_dict(torch.load('./pretrain_model_1.pk'))
#
# save_loss = etm_instance.fit(train_dataset)
#
# topics = etm_instance.get_topics(10)
# topic_coherence = etm_instance.get_topic_coherence()
# topic_diversity = etm_instance.get_topic_diversity()