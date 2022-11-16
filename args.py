### CONFIGS ###
# Data selected from:
# simulated data: 'simuA': LPCM; 'simuB': SBM; 'simuC': circle data;
# real data: 'eveques'; 'cora'.
dataset = 'cora'
model = 'GETM'

if dataset == 'simu1':  # scenario C (ARI=0.96)
    num_points = 900  # number of nodes N
    vocab_size = 558  # node features dimension (simu1: V=558, simu2: V=721, simu3: V=558)
    hidden1_dim = 512  # hidden layer dimension
    hidden2_dim = 128  # latent dimension P
    num_clusters = 4  # number of clusters K
    g_hidden_dim = 128  # graph embedding size D
    num_topics = 128  # latent dimension of topics T
    rho_size = 300  # topic embedding size L

    num_epoch = 300  # training epochs 600
    learning_rate = 5e-3  # 5e-4
    pre_lr = 0.05  # 0.01
    pre_epoch = 20  # 10

elif dataset == 'simu2':  # scenario A (ARI=0.99)
    num_points = 900  # number of nodes N
    vocab_size = 721  # node features dimension (simu1: V=558, simu2: V=721, simu3: V=558)
    hidden1_dim = 512  # hidden layer dimension
    hidden2_dim = 128  # latent dimension P
    num_clusters = 3  # number of clusters K
    g_hidden_dim = 128  # graph embedding size D
    num_topics = 128  # latent dimension of topics T
    rho_size = 300  # topic embedding size L

    num_epoch = 200  # training epochs (simu2: 600, simu3: 800)
    learning_rate = 5e-3
    pre_lr = 0.05
    pre_epoch = 20

elif dataset == 'simu3':  # scenario B (ARI=1.0)
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
    pre_epoch = 20

elif dataset == 'cora':  # TODO: NAN LOSS !!!
    num_points = 2708  # number of nodes N
    vocab_size = 25955  # 25955  # node features dimension
    hidden1_dim = 512  # hidden layer dimension
    hidden2_dim = 128  # latent dimension P
    num_clusters = 7  # number of clusters K
    g_hidden_dim = 128  # graph embedding size D
    num_topics = 128  # number of latent topics T
    rho_size = 300  # topic embedding size L

    num_epoch = 300
    learning_rate = 5e-3
    pre_lr = 0.01
    pre_epoch = 30