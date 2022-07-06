### CONFIGS ###
# Data selected from:
# simulated data: 'simuA': LPCM; 'simuB': SBM; 'simuC': circle data;
# real data: 'eveques'; 'cora'.
dataset = 'simu2'
model = 'GETM'

if dataset == 'simu1':  # scenario C
    num_points = 900  # number of nodes N
    vocab_size = 558  # node features dimension (simu1: V=558, simu2: V=721, simu3: V=558)
    hidden1_dim = 32  # hidden layer dimension
    hidden2_dim = 16  # latent dimension P
    num_clusters = 3  # number of clusters K
    g_hidden_dim = 16  # graph embedding size D
    num_topics = 2  # latent dimension of topics T
    rho_size = 300  # topic embedding size L

    num_epoch = 600  # training epochs (simu2: 600, simu3: 800)
    learning_rate = 5e-3
    pre_lr = 0.05
    pre_epoch = 20  # (simu1, 2: 5, simu3: 20)

elif dataset == 'simu2':  # scenario A
    num_points = 900  # number of nodes N
    vocab_size = 721  # node features dimension (simu1: V=558, simu2: V=721, simu3: V=558)
    hidden1_dim = 32  # hidden layer dimension
    hidden2_dim = 16  # latent dimension P
    num_clusters = 3  # number of clusters K
    g_hidden_dim = 16  # graph embedding size D
    num_topics = 3  # latent dimension of topics T
    rho_size = 300  # topic embedding size L

    num_epoch = 1000  # training epochs (simu2: 600, simu3: 800)
    learning_rate = 5e-3
    pre_lr = 0.05
    pre_epoch = 20  # (simu1, 2: 5, simu3: 20)

elif dataset == 'simu3':  # scenario B
    num_points = 900  # number of nodes N
    vocab_size = 558  # node features dimension (simu1: V=558, simu2: V=721, simu3: V=558)
    hidden1_dim = 32  # hidden layer dimension
    hidden2_dim = 16  # latent dimension P
    num_clusters = 3  # number of clusters K
    g_hidden_dim = 16  # graph embedding size D
    num_topics = 2  # latent dimension of topics T
    rho_size = 300  # topic embedding size L

    num_epoch = 600  # training epochs (simu2: 600, simu3: 800)
    learning_rate = 5e-3
    pre_lr = 0.05
    pre_epoch = 20  # (simu1, 2: 5, simu3: 20)

elif dataset == 'cora':
    num_points = 2708  # number of nodes N
    vocab_size = 25955  # node features dimension
    hidden1_dim = 128  # hidden layer dimension
    hidden2_dim = 64  # latent dimension P
    num_clusters = 7  # number of clusters K
    g_hidden_dim = 16  # graph embedding size D
    num_topics = 7  # number of latent topics T
    rho_size = 300  # topic embedding size L

    num_epoch = 600
    learning_rate = 2e-3
    pre_lr = 0.01
    pre_epoch = 100