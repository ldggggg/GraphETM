import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx

## TSNE visu ##
# z_embedded = TSNE(n_components=2, init='pca').fit_transform(mean_max.cpu().data.numpy())
# # z_embedded = A_pred.cpu().data.numpy()
# f, ax = plt.subplots(1,figsize=(15,10))
# ax.scatter(z_embedded[:,0], z_embedded[:,1], color = colC)
# plt.show()

## old PCA ##
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
out = pca.fit_transform(model.mean.cpu().data.numpy())
f, ax = plt.subplots(1,figsize=(15,10))
ax.scatter(out[:,0], out[:,1], color = colC)
ax.set_xlabel('PCA result of mean embedding of deepLSM (projection)')
plt.show()

import numpy as np

A = np.loadtxt('data/SBM/adj_SBM_1_N=300.txt')
labels = np.loadtxt('data/SBM/label_SBM_1_N=300.txt')

# load real labels
labelC = []
for idx in range(len(labels)):
    if labels[idx] == 0:
        labelC.append('#7294d4')
    elif labels[idx] == 1:
        labelC.append('#fdc765')
    else:
        labelC.append('#869f82')
import pylab as pl
def show_graph_with_labels(adjacency_matrix,labels,node_size=30,scale_pos=5.):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    for n in all_rows:
        gr.add_node(n)
    gr.add_edges_from(edges)
    pos = nx.spring_layout(gr, scale=scale_pos, seed=0)
    nx.draw_networkx_edges(gr, pos, width=1, alpha=0.5, edge_color='grey')
    # for node in gr.nodes():
    #     node_color= 'C%s'%labels[node]
    #     nx.draw_networkx_nodes(gr, pos, node_size=node_size, alpha=1, node_color=node_color)
    nx.draw(gr, pos, node_size=50, alpha=1, width=0.5, node_color=labelC, edge_color='grey')
    pl.axis('off')
    pl.tight_layout()
    pl.show()

show_graph_with_labels(A, labels)  # .todense()

f, ax = plt.subplots(1, figsize=(15, 10))
ax.scatter(model.encoder.mean.cpu().data.numpy()[:,0], model.encoder.mean.cpu().data.numpy()[:,1], color = colC)
plt.show()

####### new PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(model.mean.cpu().data.numpy())
x_pca = pca.transform(model.mean.cpu().data.numpy())

x_new = pca.inverse_transform(x_pca)
# plt.scatter(model.mean.cpu().data.numpy()[:, 0], model.mean.cpu().data.numpy()[:, 1], alpha=0.2)
plt.scatter(x_new[:, 0], x_new[:, 1], color = colC, alpha=0.8)
plt.show()

################ Sc.A ####################
import matplotlib.pyplot as plt
import numpy as np

box_1 = [0.5543, 0.5717, 0.585, 0.5539, 0.5544, 0.5399, 0.4728, 0.5673, 0.4973, 0.2413]
box_2 = [0.5958, 0.6569, 0.6159, 0.657, 0.5914, 0.6367, 0.4675, 0.6604, 0.6084, 0.6116]
box_3 = [0.6135, 0.7008, 0.5972, 0.6845, 0.6022, 0.6188, 0.4709, 0.6652, 0.6239, 0.6036]
box_4 = [0.5958, 0.6599, 0.6107, 0.657, 0.5919, 0.6419, 0.4699, 0.6531, 0.6092, 0.6075]

plt.figure(figsize=(10, 5))
plt.title('Clustering ARI', fontsize=20)
labels = 'SBM', 'ARI_pretrain_k-means', 'ARI_train_gamma', 'ARI_train_k-means'

plt.boxplot([box_1, box_2, box_3, box_4], labels=labels)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# example data A
x = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95])
# y1 = np.array([0.0549, 0.07604, 0.18923, 0.4219, 0.58272, 0.77378, 0.84894, 0.94617])  # SBM
y1 = np.array([0.07585, 0.17508, 0.30518, 0.51166, 0.68268, 0.75528, 0.85339, 0.94407])  # SBM
y2 = np.array([0.086225, 0.17251, 0.37335, 0.52959, 0.73041, 0.81963, 0.89466, 0.95942])  # our_gamma
y3 = np.array([0.09081, 0.17213, 0.37246, 0.52228, 0.71751, 0.81364, 0.88247, 0.94791])  # our+kmeans
y4 = np.array([0.03335, 0.07284, 0.20568, 0.29823, 0.4808, 0.62931, 0.80409, 0.93461])  # VGAE+kmeans
y5 = np.array([0.07791, 0.17997, 0.32436, 0.49466, 0.61271, 0.75877, 0.83286, 0.9229])  # LPCM
y6 = np.array([0.05695, 0.05624, 0.06498, 0.1016, 0.27791, 0.43503, 0.65233, 0.88366]) # ARVGA
# y7 = np.array([0.81772, 0.83777, 0.89288, 0.90238, 1.07666, 1.28358, 1.84712, 2.24534]) # no texts


# error bar values w/ different -/+ errors that
# also vary with the x-position
# lower_error1 = np.array([0.0331, 0.0457, 0.0698, 0.2404, 0.2835, 0.7378, 0.7546, 0.9045])
# upper_error1 = np.array([0.1142, 0.1083, 0.2481, 0.5533, 0.6897, 0.8291, 0.9224, 0.9801])
lower_error1 = np.array([0.0499, 0.1378, 0.2161, 0.4395, 0.5736, 0.6999, 0.8056, 0.8753])
upper_error1 = np.array([0.1393, 0.208, 0.378, 0.5991, 0.7741, 0.8096, 0.9033, 0.9801])
lower_error2 = np.array([0.0297, 0.1433, 0.3338, 0.332, 0.6895, 0.7704, 0.8386, 0.9305])
upper_error2 = np.array([0.1486, 0.1879, 0.4461, 0.6307, 0.7972, 0.866, 0.9317, 0.9801])
lower_error3 = np.array([0.0326, 0.1415, 0.3173, 0.3347, 0.6894, 0.7704, 0.8288, 0.9311])
upper_error3 = np.array([0.1741, 0.2181, 0.4219, 0.6152, 0.7876, 0.8567, 0.9215, 0.9703])
lower_error4 = np.array([0.0121, 0.012, 0.1462, 0.1506, 0.3599, 0.4727, 0.5325, 0.8846])
upper_error4 = np.array([0.0446, 0.1654, 0.2979, 0.4179, 0.6187, 0.7825, 0.8921, 0.98])
lower_error5 = np.array([0.0372, 0.1005, 0.1501, 0.4401, 0.4837, 0.6781, 0.7447, 0.8449])
upper_error5 = np.array([0.1149, 0.2805, 0.4228, 0.5638, 0.6686, 0.8138, 0.9315, 0.9506])
lower_error6 = np.array([0.0366, 0.0349, 0.046, 0.0607, 0.2104, 0.3105, 0.5253, 0.817])
upper_error6 = np.array([0.0781, 0.0748, 0.0874, 0.1571, 0.3826, 0.545, 0.7435, 0.9406])

asymmetric_error1 = [y1-lower_error1, upper_error1-y1]
asymmetric_error2 = [y2-lower_error2, upper_error2-y2]
asymmetric_error3 = [y3-lower_error3, upper_error3-y3]
asymmetric_error4 = [y4-lower_error4, upper_error4-y4]
asymmetric_error5 = [y5-lower_error5, upper_error5-y5]
asymmetric_error6 = [y6-lower_error6, upper_error6-y6]

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(x, y1, yerr=asymmetric_error1, fmt='cs--', label='SBM', marker = 's',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='c',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
ax.errorbar(x, y2, yerr=asymmetric_error2, fmt='rs--', label='DeepLPM', marker = 'v',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='r',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
# ax.errorbar(x, y3, yerr=asymmetric_error3, fmt='ys--', label='our_kmeans', marker = 'h',
#             linewidth=1,
#             elinewidth=0.6,# width of error bar line
#             ecolor='y',    # color of error bar
#             capsize=5,     # cap length for error bar
#             capthick=0.6)   # cap thickness for error bar

ax.errorbar(x, y4, yerr=asymmetric_error4, fmt='gs--', label='VGAE', marker = 'p',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='g',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
ax.errorbar(x, y5, yerr=asymmetric_error5, fmt='bs--', label='LPCM', marker = 'o',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='b',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar

ax.errorbar(x, y6, yerr=asymmetric_error6, fmt='ys--', label='ARVGA', marker = 'h',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='y',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
# ax.errorbar(x, y7, yerr=asymmetric_error7, fmt='gs--', label='deepLTRS (no texts)', marker = 's',
#             linewidth=1,
#             elinewidth=0.6,# width of error bar line
#             ecolor='g',    # color of error bar
#             capsize=5,     # cap length for error bar
#             capthick=0.6)   # cap thickness for error bar

ax.set_xlim((0.17, 0.98))
ax.set_ylabel('Clustering ARI')
#ax.set_xlabel('Effect of the sparsity for different models')
ax.set_xlabel('Rate of proximity ($\delta$)')
plt.legend(loc='upper left', fontsize=12)
plt.grid()
plt.show()
fig.savefig("C:/Users/Dingge/Desktop/results/deepLSM_ScA_new.pdf", bbox_inches='tight')


################ Sc.B ####################
import numpy as np
import matplotlib.pyplot as plt

# example data B
x = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y1 = np.array([0.37462, 0.31669, 0.52185, 0.45556, 0.37875, 0.48112, 0.5607, 0.91873])  # SBM
y11 = np.array([0.01591, 0.20605, 0.64443, 0.86948, 0.95016, 0.99624, 0.99925, 1, 1])  # SBM_Remi
# y2 = np.array([0.64265, 0.76756, 0.8892, 0.92811, 0.94018, 0.98401, 0.99778])  # our_gamma
y22 = np.array([0.00933, 0.19343, 0.85667, 0.93898, 0.984266, 0.995476, 0.998031, 0.999596, 1])  # our_gamma
# y3 = np.array([0.90081, 0.85165, 1])  # our+kmeans
y4 = np.array([0.0088, 0.02769, 0.38598, 0.59172, 0.75401, 0.85077, 0.94593, 0.98099, 0.99922])  # VGAE+kmeans
y5 = np.array([0.03535, 0.20463, 0.34548, 0.43935, 0.54027, 0.63283, 0.6201, 0.65147, 0.77354])  # LPCM
y6 = np.array([0.02344, 0.05656, 0.23899, 0.55211, 0.79245, 0.89651, 0.96444, 0.993031, 1])  # ARVGA

# error bar values w/ different -/+ errors that
# also vary with the x-position
lower_error1 = np.array([0.0015, 0.0022, 0.5321, 0.607, 0.6826, 0.9849, 0.9925, 1, 1])
upper_error1 = np.array([0.0439, 0.4964, 0.7586, 0.9778, 1, 1, 1, 1, 1])
# lower_error2 = np.array([0, 0.4926, 0.6875, 0.8207, 0.8294, 0.8636, 0.8714,0.9778])
# upper_error2 = np.array([0.3135, 0.7387, 0.8515, 0.9246, 1, 0.992, 1, 1])
lower_error22 = np.array([0, 0.0127, 0.8065, 0.9307, 0.9681, 0.9889, 0.99197, 0.99596, 1])
upper_error22 = np.array([0.0588, 0.4226, 0.8923, 0.9537, 0.9964, 1, 1, 1, 1])
lower_error4 = np.array([0.0028, 0.055, 0.2305, 0.4839, 0.7032, 0.7756, 0.9045, 0.9533, 0.9922])
upper_error4 = np.array([0.0204, 0.094, 0.5048, 0.6511, 0.82, 0.9351, 0.9783, 1, 1])
lower_error5 = np.array([0.00695, 0.094, 0.2945, 0.3683, 0.4621, 0.5755, 0.5714, 0.5847, 0.6029])
upper_error5 = np.array([0.06775, 0.2655, 0.3803, 0.4795, 0.595, 0.674, 0.668, 0.6887, 1])
lower_error6 = np.array([0.01106, 0.0448, 0.135, 0.4149, 0.6712, 0.829, 0.9312, 0.985, 1])
upper_error6 = np.array([0.03572, 0.0834, 0.3472, 0.7143, 0.8831, 0.9521, 0.9883, 1, 1])

asymmetric_error1 = [y11-lower_error1, upper_error1-y11]
# asymmetric_error2 = [y2-lower_error2, upper_error2-y2]
asymmetric_error22 = [y22-lower_error22, upper_error22-y22]
asymmetric_error4 = [y4-lower_error4, upper_error4-y4]
asymmetric_error5 = [y5-lower_error5, upper_error5-y5]
asymmetric_error6 = [y6-lower_error6, upper_error6-y6]

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(x, y11, yerr=asymmetric_error1, fmt='cs--', label='SBM', marker = 's',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='c',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
ax.errorbar(x, y22, yerr=asymmetric_error22, fmt='rs--', label='DeepLPM', marker = 'v',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='r',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
# ax.errorbar(x, y3, yerr=asymmetric_error3, fmt='ys--', label='our_kmeans', marker = 'h',
#             linewidth=1,
#             elinewidth=0.6,# width of error bar line
#             ecolor='y',    # color of error bar
#             capsize=5,     # cap length for error bar
#             capthick=0.6)   # cap thickness for error bar

ax.errorbar(x, y4, yerr=asymmetric_error4, fmt='gs--', label='VGAE', marker = 'p',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='g',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
ax.errorbar(x, y5, yerr=asymmetric_error5, fmt='bs--', label='LPCM', marker = 'o',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='b',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
ax.errorbar(x, y6, yerr=asymmetric_error6, fmt='ys--', label='ARVGA', marker = 'h',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='y',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar

ax.set_xlim((0.17, 1.03))
ax.set_ylabel('Clustering ARI')  # , fontsize=18
#ax.set_xlabel('Effect of the sparsity for different models')
ax.set_xlabel('Rate of proximity ($\delta^{\'}$)')
plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.show()
fig.savefig("C:/Users/Dingge/Desktop/results/deepLSM_ScB_new_0.2.pdf", bbox_inches='tight')


################ SBM ####################
import numpy as np
import matplotlib.pyplot as plt

# example data A
x = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95])
y1 = np.array([0.07853, 0.14247, 0.25245, 0.39276, 0.60325, 0.68671, 0.7402, 0.9167])  # SBM
y2 = np.array([0.07873, 0.14291, 0.25245, 0.43974, 0.60325, 0.74639, 0.82114, 0.9167])  # our_gamma
# y3 = np.array([0.90081, 0.85165, 1])  # our+kmeans
y4 = np.array([0.08147, 0.14857, 0.25101, 0.43974, 0.60325, 0.74639, 0.82114, 0.9167])  # VGAE+kmeans
y5 = np.array([0.08192, 0.14892, 0.25178, 0.43974, 0.60325, 0.74639, 0.82114, 0.9167])  # LPCM

# error bar values w/ different -/+ errors that
# also vary with the x-position
lower_error1 = np.array([0.0376, 0.0527, 0.1855, 0, 0.4699, 0, 0, 0.8215])
upper_error1 = np.array([0.1321, 0.2479, 0.3501, 0.6175, 0.6604, 0.8336, 0.8707, 0.9602])
lower_error2 = np.array([0.0376, 0.0527, 0.1855, 0.3345, 0.4699, 0.5968, 0.6823, 0.8215])
upper_error2 = np.array([0.1321, 0.2479, 0.3501, 0.6175, 0.6604, 0.8336, 0.8707, 0.9602])
# lower_error3 = np.array([0.6189, 0.6228, 1])
# upper_error3 = np.array([0.9916, 1, 1])
lower_error4 = np.array([0.0403, 0.0527, 0.1855, 0.3345, 0.4699, 0.5968, 0.6823, 0.8215])
upper_error4 = np.array([0.1321, 0.2547, 0.3501, 0.6175, 0.6604, 0.8336, 0.8707, 0.9602])
lower_error5 = np.array([0.0403, 0.0558, 0.1855, 0.3345, 0.4699, 0.5968, 0.6823, 0.8215])
upper_error5 = np.array([0.1321, 0.2547, 0.3501, 0.6175, 0.6604, 0.8336, 0.8707, 0.9602])

asymmetric_error1 = [y1-lower_error1, upper_error1-y1]
asymmetric_error2 = [y2-lower_error2, upper_error2-y2]
# asymmetric_error3 = [y3-lower_error3, upper_error3-y3]
asymmetric_error4 = [y4-lower_error4, upper_error4-y4]
asymmetric_error5 = [y5-lower_error5, upper_error5-y5]

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(x, y1, yerr=asymmetric_error1, fmt='bs--', label='SBM_ini_1', marker = 's',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='b',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
ax.errorbar(x, y2, yerr=asymmetric_error2, fmt='cs--', label='SBM_ini_10', marker = 'v',
            linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='c',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
# ax.errorbar(x, y3, yerr=asymmetric_error3, fmt='ys--', label='our_kmeans', marker = 'h',
#             linewidth=1,
#             elinewidth=0.6,# width of error bar line
#             ecolor='y',    # color of error bar
#             capsize=5,     # cap length for error bar
#             capthick=0.6)   # cap thickness for error bar

ax.errorbar(x, y4, yerr=asymmetric_error4, fmt='gs--', label='SBM_ini_100', linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='g',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar
ax.errorbar(x, y5, yerr=asymmetric_error5, fmt='rs--', label='SBM_ini_1000', linewidth=1,
            elinewidth=0.6,# width of error bar line
            ecolor='r',    # color of error bar
            capsize=5,     # cap length for error bar
            capthick=0.6)   # cap thickness for error bar

ax.set_xlim((0.15, 1.0))
ax.set_ylabel('Clustering ARI', fontsize=13)
#ax.set_xlabel('Effect of the sparsity for different models')
ax.set_xlabel('Number of initializations in SBM', fontsize=13)
plt.legend(loc='upper left', fontsize=12)
plt.show()
fig.savefig("C:/Users/Dingge/Desktop/results/diff_SBM_ini.pdf", bbox_inches='tight')


###################################### Diff Dim #########################################
import numpy as np
import matplotlib.pyplot as plt
import math

# example data B
x = np.array([2, 4, 8, 16, 32])
y1 = np.array([0.69734, 0.73519, 0.79551, 0.82758, 0.66264])  # ARI
y2 = np.array([44823.19384, 44733.6414, 44611.65352, 44528.8, 44670.0782])  # Loss

lower_error1 = np.array([0.5814, 0.5838, 0.7014, 0.699, 0])
upper_error1 = np.array([0.8441, 0.8287, 0.8415, 0.8593, 0.9002])
lower_error2 = np.array([44400.2344, 44381.48, 44210.32, 44204.6211, 44200.785])
upper_error2 = np.array([45105.141, 44983.81, 44958.84, 44855.36, 45304.75])
asymmetric_error1 = [y1-lower_error1, upper_error1-y1]
asymmetric_error2 = [y2-lower_error2, upper_error2-y2]

f, ax = plt.subplots(1, figsize=(12, 5))
plt.subplot(122)
# plt.errorbar(x, y1, yerr=asymmetric_error1, fmt='bs--', color='blue', linewidth=1,
#             elinewidth=0.6,# width of error bar line
#             ecolor='b',    # color of error bar
#             capsize=5,     # cap length for error bar
#             capthick=0.6)
plt.errorbar(x, y1, fmt='bs--', color='blue')
plt.xlabel('Latent dimension')
plt.ylabel('Clustering ARI')

plt.subplot(121)
# plt.errorbar(x, y2, yerr=asymmetric_error2, fmt='rs--', color='red', linewidth=1,
#             elinewidth=0.6,# width of error bar line
#             ecolor='r',    # color of error bar
#             capsize=5,     # cap length for error bar
#             capthick=0.6)
plt.errorbar(x, y2, fmt='rs--', color='pink')
plt.xlabel('Latent dimension')
plt.ylabel('Training loss')
# plt.show()
f.savefig("C:/Users/Dingge/Desktop/results/diff_dim.pdf", bbox_inches='tight')

################  boxplot ################
import numpy as np
import matplotlib.pyplot as plt

# example data B
x = [2,4,8,16,32]
y1 = np.array([0.6548,0.6051,0.5814,0.6728,0.7711,0.6309,0.7022,0.7852,0.8441,0.7258])
y2 = np.array([0.7337,0.6494,0.7761,0.7841,0.7841,0.6723,0.7446,0.8456,0.7733,0.6321])
y3 = np.array([0.6534,0.6506,0.7005,0.7216,0.7011,0.6383,0.7418,0.3858,0.7271,0.6408])
y4 = np.array([0.6561,0.6698,0.7265,0.7188,0.7034,0.5188,0.6736,0.4567,0.7439,0.6206])
y5 = np.array([0.6387,0.5483,0.6729,0.6826,0.7137,0.5156,0.7762,0.4432,0.6512,0.5942])
data = [y1, y2, y3, y4, y5]

y11 = np.array([50310.22,50321.5,49657.99,49621.32,49616.3,50178.47,50785.09,53102.28,49149.32,49429.33])
y21 = np.array([50203.04,50237.75,49530.16,49568.7,49483.81,50195.62,50780.92,53102.82,49128.41,49368.72])
y31 = np.array([50241.68,50219.95,49524.45,49588.58,49600.98,50222.16,50712.48,53044.77,49114.86,49391.36])
y41 = np.array([50324.78,50406.38,49527.69,49548.41,49605.71,50241.5,50709.2,53069.63,49092.84,49343.54])
y51 = np.array([50266.04,50197.46,49551.02,49563.58,49565.38,50138.02,50738.82,53083.24,49066.86,49336.11])
data1 = [y11, y21, y31, y41, y51]

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplot(121)
bplot = plt.boxplot(data, labels=x, sym='o',vert=True, patch_artist=True)

colors = ['lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue']
for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.ylabel('Clustering ARI', fontsize=18)
plt.xlabel('Different number of clusters', fontsize=18)


plt.subplot(122)
bplot = plt.boxplot(data1, labels=x, sym='o',vert=True, patch_artist=True)

colors = ['pink', 'pink', 'pink', 'pink', 'pink']
for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.ylabel('Clustering ARI', fontsize=18)
plt.xlabel('Different number of clusters', fontsize=18)

###### visu data C when P=2 #######
out2 = model.encoder.mean.cpu().data.numpy()
f, ax = plt.subplots(1, figsize=(10, 10))
ax.scatter(out2[:, 0], out2[:, 1], color=labelC)
# ax.scatter(mean[:, 0], mean[:, 1], color='black', s=50)
ax.set_title('Latent embeddings of DeepLPM', fontsize=18)
plt.show()
f.savefig("C:/Users/Dingge/Desktop/results/emb_DeepLPM_C300.pdf", bbox_inches='tight')


###################################### Diff Clusters A #########################################
import numpy as np
import matplotlib.pyplot as plt
import math

# example data A
x = np.array([2, 3, 4, 5, 6])
y1 = np.array([0.43684,0.73953,0.6561,0.64882,0.62366])  # ARI
y2 = np.array([50217.182,50159.995,50166.127,50186.968,50150.653])  # Loss
loss1 = np.log(math.factorial(2))
loss2 = np.log(math.factorial(3))
loss3 = np.log(math.factorial(4))
loss4 = np.log(math.factorial(5))
loss5 = np.log(math.factorial(6))
y3 = np.array([50217.182+loss1,50159.995+loss2,50166.127+loss3,50186.968+loss4,50150.653+loss5])  # Loss

lower_error1 = np.array([0.2713, 0.6321, 0.3858, 0.4567, 0.4432])
upper_error1 = np.array([0.7766, 0.8456, 0.7418, 0.7439, 0.7762])
lower_error2 = np.array([49149.32, 49128.41, 49114.86, 49092.84, 49066.86])
upper_error2 = np.array([53102.28, 53102.82, 53044.77, 53069.63, 53083.24])
asymmetric_error1 = [y1-lower_error1, upper_error1-y1]
asymmetric_error2 = [y2-lower_error2, upper_error2-y2]

f, ax = plt.subplots(1, figsize=(9, 5))
plt.subplot(131)
# plt.errorbar(x, y1, yerr=asymmetric_error1, fmt='bs--', color='blue', linewidth=1,
#             elinewidth=0.6,# width of error bar line
#             ecolor='b',    # color of error bar
#             capsize=5,     # cap length for error bar
#             capthick=0.6)
plt.errorbar(x, y1, fmt='bs--', color='blue')
plt.xlabel('Different number of clusters', fontsize=18)
plt.ylabel('Clustering ARI', fontsize=18)

plt.subplot(132)
# plt.errorbar(x, y2, yerr=asymmetric_error2, fmt='rs--', color='red', linewidth=1,
#             elinewidth=0.6,# width of error bar line
#             ecolor='r',    # color of error bar
#             capsize=5,     # cap length for error bar
#             capthick=0.6)
plt.errorbar(x, y2, fmt='rs--', color='red')
plt.xlabel('Different number of clusters', fontsize=18)
plt.ylabel('Training loss', fontsize=18)

plt.subplot(133)
plt.errorbar(x, y3, fmt='rs--', color='pink')
plt.xlabel('Different number of clusters', fontsize=18)
plt.ylabel('Training loss + lnK!', fontsize=18)
# plt.show()
f.savefig("C:/Users/Dingge/Desktop/results/diff_clus.pdf", bbox_inches='tight')


###################################### Diff Clusters B #########################################
import numpy as np
import matplotlib.pyplot as plt
import math

# example data B
x = np.array([2, 3, 4, 5, 6])
y1 = np.array([0.77998,0.82922,0.7794,0.80142,0.82606])  # ARI
y2 = np.array([44411.168,44375.3126,44379.1646,44388.0588,44383.4688])  # Loss
loss1 = np.log(math.factorial(2))
loss2 = np.log(math.factorial(3))
loss3 = np.log(math.factorial(4))
loss4 = np.log(math.factorial(5))
loss5 = np.log(math.factorial(6))
y3 = np.array([44411.168+loss1,44375.3126+loss2,44379.1646+loss3,44388.0588+loss4,44383.4688+loss5])  # Loss

f, ax = plt.subplots(1, figsize=(12, 5))
# plt.subplot(131)
# # plt.errorbar(x, y1, yerr=asymmetric_error1, fmt='bs--', color='blue', linewidth=1,
# #             elinewidth=0.6,# width of error bar line
# #             ecolor='b',    # color of error bar
# #             capsize=5,     # cap length for error bar
# #             capthick=0.6)
# plt.errorbar(x, y1, fmt='bs--', color='blue')
# plt.xlabel('Different number of clusters', fontsize=18)
# plt.ylabel('Clustering ARI', fontsize=18)

# plt.subplot(121)
# plt.errorbar(x, y2, yerr=asymmetric_error2, fmt='rs--', color='red', linewidth=1,
#             elinewidth=0.6,# width of error bar line
#             ecolor='r',    # color of error bar
#             capsize=5,     # cap length for error bar
#             capthick=0.6)
plt.errorbar(x, y2, fmt='rs--', color='red')
plt.xlabel('Number of clusters')
plt.ylabel('Training loss')

# plt.subplot(122)
# plt.errorbar(x, y3, fmt='rs--', color='pink')
# plt.xlabel('Different number of clusters', fontsize=15)
# plt.ylabel('Training loss + lnK!', fontsize=15)
# plt.show()
f.savefig("C:/Users/Dingge/Desktop/results/diff_clus_B.pdf", bbox_inches='tight')


###################################### Diff Clusters Eveques #########################################
import numpy as np
import matplotlib.pyplot as plt
import math

# Eveques
x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
# y1 = np.array([49700.816,50383.957,49730.63,49746.074,49308.95,49097.1,47674.953,47907.89,48408.508])  # Loss
# y1 = np.array([111318.23,97913.37,97400.1,91841.89,110993.375,103555.516,111800.48,103866.34,104907.41])  # new Loss
y1 = np.array([56128.19,56546.16,53937.758,53600.926,53752.582,53350.586,53574.434,52744.066,53094.164,53194.35,
               52290.66,51933.543,51968.02,52323.63,53020.793,52060.957])
y2 = np.array([69242.39,66538.74,68233.836,66352.37,66466.32,65593.84,64038.84,62472.285,64373.72])

# fact = np.zeros(9)
# for i in range(9):
#     fact[i] = np.log(math.factorial(i+2))
# y2 = y1 + fact  # Loss+LnK!

f, ax = plt.subplots(1, figsize=(12, 5))
# plt.subplot(121)
plt.errorbar(x, y2, fmt='rs--', color='red')
plt.xlabel('Different number of clusters', fontsize=15)
plt.ylabel('Training loss', fontsize=15)

# plt.subplot(122)
# plt.errorbar(x, y2, fmt='rs--', color='pink')
# plt.xlabel('Different number of clusters', fontsize=15)
# plt.ylabel('Training loss + lnK!', fontsize=15)
# plt.show()
f.savefig("C:/Users/Dingge/Desktop/results/diff_clus_eveques_k=9.pdf", bbox_inches='tight')

################  boxplot ################
import numpy as np
import matplotlib.pyplot as plt

# example data A
x = [2, 3, 4, 5, 6]
y1 = np.array([0.4052,0.2713,0.3928,0.3631,0.3828,0.6438,0.373,0.7766,0.3254,0.4344])
y2 = np.array([0.7337,0.6494,0.7761,0.7841,0.7841,0.6723,0.7446,0.8456,0.7733,0.6321])
y3 = np.array([0.6534,0.6506,0.7005,0.7216,0.7011,0.6383,0.7418,0.3858,0.7271,0.6408])
y4 = np.array([0.6561,0.6698,0.7265,0.7188,0.7034,0.5188,0.6736,0.4567,0.7439,0.6206])
y5 = np.array([0.6387,0.5483,0.6729,0.6826,0.7137,0.5156,0.7762,0.4432,0.6512,0.5942])
data = [y1, y2, y3, y4, y5]

y11 = np.array([50310.22,50321.5,49657.99,49621.32,49616.3,50178.47,50785.09,53102.28,49149.32,49429.33])
y21 = np.array([50203.04,50237.75,49530.16,49568.7,49483.81,50195.62,50780.92,53102.82,49128.41,49368.72])
y31 = np.array([50241.68,50219.95,49524.45,49588.58,49600.98,50222.16,50712.48,53044.77,49114.86,49391.36])
y41 = np.array([50324.78,50406.38,49527.69,49548.41,49605.71,50241.5,50709.2,53069.63,49092.84,49343.54])
y51 = np.array([50266.04,50197.46,49551.02,49563.58,49565.38,50138.02,50738.82,53083.24,49066.86,49336.11])
data1 = [y11, y21, y31, y41, y51]

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplot(121)
bplot = plt.boxplot(data, labels=x, sym='o',vert=True, patch_artist=True)

colors = ['lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue']
for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.ylabel('Clustering ARI', fontsize=18)
plt.xlabel('Different number of clusters', fontsize=18)


plt.subplot(122)
bplot = plt.boxplot(data1, labels=x, sym='o',vert=True, patch_artist=True)

colors = ['pink', 'pink', 'pink', 'pink', 'pink']
for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.ylabel('Clustering ARI', fontsize=18)
plt.xlabel('Different number of clusters', fontsize=18)

plt.show()
fig.savefig("C:/Users/Dingge/Desktop/results/diff_clus.pdf", bbox_inches='tight')

################ Sc.C ####################
import numpy as np
import matplotlib.pyplot as plt

# example data C
x = ['SBM', 'DeepLPM (P=16)', 'VGAE_kmeans (P=8)', 'LPCM']
y1 = np.array([0.4422,0.4439,0.4446,0.442,0.4424,0.4428,0.442,0.4422])  # SBM
y2 = np.array([0.5791,0.5659,0.6511,0.6466,0.6466,0.6119,0.6466,0.6193])  # our_gamma
y3 = np.array([0.6078,0.5683,0.5904,0.605,0.6296,0.6379,0.6421,0.5533])  # VGAE+kmeans
y4 = np.array([0.1122,0.3178,0.5021,0.4178,0.1678,0.5021,0.5079,0.0625])  # LPCM
data = [y1, y2, y3, y4]

fig, ax = plt.subplots(figsize=(8, 5))
bplot = ax.boxplot(data, labels=x, sym='o',vert=True, patch_artist=True)

colors = ['cyan', 'pink', 'lightgreen', 'lightblue']
for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

ax.set_ylabel('Clustering ARI', fontsize=18)
plt.show()
fig.savefig("C:/Users/Dingge/Desktop/results/deepLSM_ScC.pdf", bbox_inches='tight')

import sknetwork as skn
from IPython.display import SVG
from sknetwork.data import karate_club, painters, movie_actor, load_netset
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph


###############################  GETM ####################################
import numpy as np
import matplotlib.pyplot as plt
import math

# scenario C
x = np.array([2, 3, 4, 5, 6, 7])
y1 = np.array([996.51166,991.25996,995.91842,998.4835833,999.81085,999.9620143])  # Loss
f, ax = plt.subplots(1, figsize=(9, 12))
plt.errorbar(x, y1, fmt='rs--', color='red')
plt.xlabel('Different number of clusters', fontsize=18)
plt.ylabel('Training loss (-ELBO)', fontsize=18)
plt.show()
f.savefig("data/diff_clus.pdf", bbox_inches='tight')