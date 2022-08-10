import numpy as np
import pickle
from scipy.stats import bernoulli

#import nltk
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
#from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from random import sample, shuffle

# loading texts and removing punctuation
tokenizer = RegexpTokenizer(r'\w+')
A = tokenizer.tokenize(open('C:/Users/Dingge/Doctoral_projets/Pytorch/msgA.txt').read())
B = tokenizer.tokenize(open('C:/Users/Dingge/Doctoral_projets/Pytorch/msgB.txt', encoding="utf8").read())
C = tokenizer.tokenize(open('C:/Users/Dingge/Doctoral_projets/Pytorch/msgC.txt').read())
D = tokenizer.tokenize(open('C:/Users/Dingge/Doctoral_projets/Pytorch/msgD.txt').read())
# cora = tokenizer.tokenize(open("data/Cora_enrich/texts.txt").read())

# Turning everything to lowercase
A = [idx.lower() for idx in A]
B = [idx.lower() for idx in B]
C = [idx.lower() for idx in C]
D = [idx.lower() for idx in D]
# cora = [idx.lower() for idx in cora]

# removing stop words
stop_words = set(stopwords.words('english'))
A = [w for w in A if not w in stop_words]
B = [w for w in B if not w in stop_words]
C = [w for w in C if not w in stop_words]
D = [w for w in D if not w in stop_words]
# cora = [w for w in cora if not w in stop_words]

################################################ Cora ############################################################
# creating a dictionary from the above texts
corpus = [cora]
dct = Dictionary(corpus)
dct.save("C:/Users/Dingge/Documents/GitHub/GETM/data/Cora_enrich/dic_cora.pkl")

######################################################Sc.A########################################################
# creating a dictionary from the above texts
corpus = [A,B,A]
dct = Dictionary(corpus)
dct.save("C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/dic_BBC_1_N=300.pkl")
# dct[72]
# dct.token2id["baby"]
# dct = pickle.load(open('dic_BBC.pkl','rb'))
dctn = dct.token2id

def create_simu1(N, K):

    Pi = np.zeros((K, K))
    a = 0.8
    b = 0.2

    Pi[0,0] = a
    Pi[0,1] = a
    Pi[0,2] = b
    Pi[1,0] = a
    Pi[1,1] = a
    Pi[1,2] = b
    Pi[2,0] = b
    Pi[2,1] = b
    Pi[2,2] = b

    # clusters
    Rho = [0.3, 0.3, 0.4]
    # N=5
    c = np.random.multinomial(1, Rho, size=N)
    c = np.argmax(c, axis=1)

    # c = np.loadtxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_SBM_1_N=300.txt', dtype=int)

    A = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            prob = Pi[c[i], c[j]]
            A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)

    docs = []
    for i in range(N):
        pos = c[i]
        msg = []
        Nw = int(np.random.normal(100, 5))  # number of words picked in each text
        sampled_pos = sample(range(len(corpus[pos])), Nw)  # select words from corpus randomly and get their indices
        for idz in sampled_pos:
            msg.append(corpus[pos][idz])
        docs.append(msg)

    label = []
    for idx in range(len(c)):
        if c[idx] == 0:
            label.append('#7294d4')
        elif c[idx] == 1:
            label.append('#fdc765')
        else:
            label.append('#869f82')

    label_text = []
    for idx in range(len(c)):
        if c[idx] == 0:
            label_text.append(0)  # A
        elif c[idx] == 1:
            label_text.append(1)  # B
        else:
            label_text.append(0)  # A

    # Saving data
    with open('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/sim_docs_SBM_1_N=300', 'wb') as fp:
        pickle.dump(docs, fp)
    np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/adj_SBM_1_N=300.txt', A)
    np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_SBM_1_N=300.txt', c)
    np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_text_SBM_1_N=300.txt', label_text)

create_simu1(300, 3)

with open ('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/sim_docs_SBM_1', 'rb') as fp:
    list = pickle.load(fp)
    new_list = [" ".join(x) for x in list]

######################################################Sc.B########################################################
# creating a dictionary from the above texts
corpus2 = [A,B,C]
dct2 = Dictionary(corpus2)
dct2.save("C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/dic_BBC_2.pkl")
# dct[72]
# dct.token2id["baby"]
# dct = pickle.load(open('dic_BBC.pkl','rb'))
dctn2 = dct2.token2id

def create_simu2(N, K):

    Pi = np.zeros((K, K))
    a = 0.8
    b = 0.2

    Pi[0,0] = a
    Pi[0,1] = a
    Pi[0,2] = b
    Pi[1,0] = a
    Pi[1,1] = a
    Pi[1,2] = b
    Pi[2,0] = b
    Pi[2,1] = b
    Pi[2,2] = b

    Rho = [0.3, 0.3, 0.4]
    # N=5
    c = np.random.multinomial(1, Rho, size=N)
    c = np.argmax(c, axis=1)

    c = np.loadtxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_SBM_2.txt', dtype=int)

    A = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            prob = Pi[c[i], c[j]]
            A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)

    docs = []
    for i in range(N):
        pos = c[i]
        msg = []
        Nw = int(np.random.normal(100, 5))  # number of words picked in each text
        sampled_pos = sample(range(len(corpus2[pos])), Nw)
        for idz in sampled_pos:
            msg.append(corpus2[pos][idz])
        docs.append(msg)

    label = []
    for idx in range(len(c)):
        if c[idx] == 0:
            label.append('#7294d4')
        elif c[idx] == 1:
            label.append('#fdc765')
        else:
            label.append('#869f82')

    label_text = []
    for idx in range(len(c)):
        if c[idx] == 0:
            label_text.append(0)  # A
        elif c[idx] == 1:
            label_text.append(1)  # B
        else:
            label_text.append(2)  # C

    # Saving data
    with open('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/sim_docs_SBM_2', 'wb') as fp:
        pickle.dump(docs, fp)
    np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/adj_SBM_2.txt', A)
    np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_SBM_2.txt', c)
    np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_text_SBM_2.txt', label_text)

create_simu2(900, 3)

######################################################Sc.C########################################################
# creating a dictionary from the above texts
corpus3 = [A,B,A]
dct3 = Dictionary(corpus3)
dct3.save("C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/dic_BBC_3.pkl")
# dct[72]
# dct.token2id["baby"]
# dct = pickle.load(open('dic_BBC.pkl','rb'))
dctn3 = dct3.token2id

def create_simu3(N, K):

    Pi = np.zeros((K, K))
    a = 0.7
    b = 0.3

    Pi[0,0] = a
    Pi[0,1] = b
    Pi[0,2] = b
    Pi[1,0] = b
    Pi[1,1] = a
    Pi[1,2] = b
    Pi[2,0] = b
    Pi[2,1] = b
    Pi[2,2] = a

    # Rho = [0.3, 0.3, 0.4]
    # # N=5
    # c = np.random.multinomial(1, Rho, size=N)
    # c = np.argmax(c, axis=1)

    c = np.loadtxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_SBM_3.txt', dtype=int)

    A = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            prob = Pi[c[i], c[j]]
            A[i,j] = A[j,i] = bernoulli.rvs(prob, loc=0, size=1)

    # docs = []
    # for i in range(N):
    #     pos = c[i]
    #     msg = []
    #     Nw = int(np.random.normal(100, 5))  # number of words picked in each text
    #     sampled_pos = sample(range(len(corpus3[pos])), Nw)
    #     for idz in sampled_pos:
    #         msg.append(corpus3[pos][idz])
    #     docs.append(msg)

    # label = []
    # for idx in range(len(c)):
    #     if c[idx] == 0:
    #         label.append('#7294d4')
    #     elif c[idx] == 1:
    #         label.append('#fdc765')
    #     else:
    #         label.append('#869f82')
    #
    # label_text = []
    # for idx in range(len(c)):
    #     if c[idx] == 0:
    #         label_text.append(0)  # A
    #     elif c[idx] == 1:
    #         label_text.append(1)  # B
    #     else:
    #         label_text.append(0)  # A

    # Saving data
    # with open('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/sim_docs_SBM_3', 'wb') as fp:
    #     pickle.dump(docs, fp)
    np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/adj_SBM_3_pi=0.7.txt', A)
    # np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_SBM_3.txt', c)
    # np.savetxt('C:/Users/Dingge/Documents/GitHub/GETM/data/SBM/label_text_SBM_3.txt', label_text)

create_simu3(900, 3)