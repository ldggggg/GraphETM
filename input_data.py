'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import args
import pickle
import embedding
from gensim.corpora import Dictionary

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    if dataset == 'cora':
        adjacency = np.loadtxt("data/Cora_enrich/cora_adj.txt")
        labels = np.loadtxt("data/Cora_enrich/cora_labels_int.txt")

        ########################## docs and vocab #################################
        dct = pickle.load(open('data/Cora_enrich/dic_cora.pkl', 'rb'))
        dctn = dct.token2id
        V = len(dctn)

        with open("data/Cora_enrich/texts.txt", "r") as f:
            txts = f.readlines()
            txts = [x.strip("\n") for x in txts]
        docs = [x.split(' ') for x in txts]

        # num version of docs
        ndocs = []
        for doc in range(len(docs)):
            tmp = []
            for word in docs[doc]:
                tmp.append(dctn[word])
            ndocs.append(tmp)

        # complete dtm
        cdtm = []
        for idx in range(len(ndocs)):
            cdtm.append(np.bincount(ndocs[idx], minlength=V))
        features = np.asarray(cdtm, dtype='float32')

        ############################## embeddings ###################################
        with open('data/Cora_enrich/word2vec_model_1.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        vectors = dict()
        key = embeddings.index2word
        value = embeddings.vectors
        for i in range(args.vocab_size):
            vectors[key[i]] = value[i]
        model_embeddings = np.zeros((V, 300))
        for i, word in enumerate(dctn):
            # print(i, word)
            model_embeddings[i] = vectors[word]

    elif dataset == 'simu2':
        adjacency = np.loadtxt("data/SBM/adj_SBM_2.txt")
        labels = np.loadtxt("data/SBM/label_SBM_2.txt")

        ############## loading and manipulatind docs and vocabulary  ###############
        dct = pickle.load(open('data/SBM/dic_BBC_2.pkl', 'rb'))
        dctn = dct.token2id
        V = len(dctn)

        with open('data/SBM/sim_docs_SBM_2', 'rb') as fp:
            docs = pickle.load(fp)

        # num version of docs
        ndocs = []
        for doc in range(len(docs)):
            tmp = []
            for word in docs[doc]:
                tmp.append(dctn[word])
            ndocs.append(tmp)

        # complete dtm
        cdtm = []
        for idx in range(len(ndocs)):
            cdtm.append(np.bincount(ndocs[idx], minlength=V))
        features = np.asarray(cdtm, dtype='float32')

        ####################### Training word2vec embeddings ######################
        # documents = [" ".join(x) for x in docs]
        # embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(documents)
        # # Save embeddings for rho
        # with open('data/SBM/word2vec_SBM_2.pkl', 'wb') as fout:
        #     pickle.dump((embeddings_mapping), fout)

        with open('data/SBM/word2vec_SBM_2.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        vectors = dict()
        key = embeddings.index2word
        value = embeddings.vectors
        for i in range(args.vocab_size):
            vectors[key[i]] = value[i]
        model_embeddings = np.zeros((V, 300))
        for i, word in enumerate(dctn):
            # print(i, word)
            model_embeddings[i] = vectors[word]

    return features, adjacency, labels, model_embeddings

def get_topics(beta, top_n_words=10):
    """
    Gets topics. By default, returns the 10 most relevant terms for each topic.
    Parameters:
    ===
        top_n_words (int): number of top words per topic to return
    Returns:
    ===
        list of str: topic list
    """
    if args.dataset == 'simu2':
        dct = pickle.load(open('data/SBM/dic_BBC_2.pkl', 'rb'))
    elif args.dataset == 'cora':
        dct = pickle.load(open('data/Cora_enrich/dic_cora.pkl', 'rb'))

    topics = []
    for k in range(args.num_topics):
        beta_k = beta[k]
        top_words = list(beta_k.argsort()[-top_n_words:][::-1])
        topic_words = [dct[a] for a in top_words]
        topics.append(topic_words)
    return topics


def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for document in data:
            # FIXME: 'if' for original article's code, 'else' for updated
            doc = document.squeeze(0) if document.shape[0] == 1 else document

            if wi in doc:
                D_wi += 1
        return D_wi

    D_wj = 0
    D_wi_wj = 0
    for document in data:
        # FIXME: 'if' for original article's code, 'else' for updated version
        doc = document.squeeze(0) if document.shape[0] == 1 else document

        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj


def get_topic_coherence(beta, data, top_n=10):
    D = len(data)  # number of docs...data is list of documents
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        beta_top_n = list(beta[k].argsort()[-top_n:][::-1])
        TC_k = 0
        counter = 0
        for i, word in enumerate(beta_top_n):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(beta_top_n) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(
                    data, word, beta_top_n[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) -
                                    2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                # update tmp:
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp
        TC.append(TC_k)
    TC = np.mean(TC) / counter
    return TC


