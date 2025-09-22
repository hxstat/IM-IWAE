import numpy as np
import pandas as pd
import scipy as sp
import os
import sys
sys.path.append(os.getcwd())
from ..IM-IWAE import newModel
from .. import trainer
from .. import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# ---- data settings
name = '/tmp/yahoo/task01/best'
n_hidden = 128
n_samples = 20
max_iter = 100
batch_size = 100
L = 10

# ---- choose the missing model
mprocess = 'linear'

# ---- number of runs
runs = 5
RMSE_model = []


train_txt = pd.read_csv(r'ydata-ymusic-rating-study-v1_0-train.txt', sep = '\t', names = ['user_id', 'song_id', 'rating'])

train_sparse = sp.sparse.csr_matrix((train_txt['rating'], (train_txt['user_id']-1, train_txt['song_id']-1)))

train_mat = train_sparse.todense()
train_mat = train_mat/5.0


test_txt = pd.read_csv(r'ydata-ymusic-rating-study-v1_0-test.txt', sep = '\t', names = ['user_id', 'song_id', 'rating'])

test_sparse = sp.sparse.csr_matrix((test_txt['rating'], (test_txt['user_id']-1, test_txt['song_id']-1)))

test_mat = test_sparse.todense()

X_test = np.array(test_mat)
R_test = 1-(X_test ==0).astype(float)

Xz = np.array(train_mat)
R = 1-(Xz ==0).astype(float)
Xz_df = pd.DataFrame(Xz)
Xnan_df = Xz_df.replace(0, np.nan, inplace=False)
Xnan = np.array(Xnan_df)


for run in range(runs):

    N, D = Xz.shape
    N_test = X_test.shape[0]

    
    np.random.seed(run)
    S = np.array(~np.isnan(Xnan), dtype=float)

    Xtrainnan = Xnan
    Xtrainz = Xz

    # validation data 
    Xvalnan = Xnan
    Xvalz = Xz


    # ---------------------- #
    # ---- fit newModel---- #
    # ---------------------- #
    model = newModel(Xtrainnan, Xvalnan, n_latent = 20,n_latent_tilda = 1, code_size = 20, complexity = "low",decoder_structure ="ma", n_samples=n_samples, n_hidden=n_hidden,sig2 = 0.02,
                        permutation_invariance=True, name=name)

    # ---- do the training
    trainer0.train(model, batch_size=batch_size, max_iter=max_iter, name=name + 'model')

    # ---- find imputation RMSE
    Xrec = 5.0*utils0.not_imputationRMSE(model, Xz, Xz, Xnan, S, L)[1]


    RMSE_model.append((np.sqrt(np.sum((X_test - R_test*(Xrec[:N_test,:])) ** 2) / np.sum(R_test))))
    
    #print(RMSE_model_ez)
    print(RMSE_model)

    print('MODEL {0:.5f}'.format(RMSE_model[-1]))

Xrec_df = pd.DataFrame(Xrec)
print("RMSE_model = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_model), np.std(RMSE_model)))

