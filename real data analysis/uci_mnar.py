import numpy as np
import pandas as pd
import os
import sys
import random
sys.path.append(os.getcwd())
from ..NIM-IWAE import newModel
from ..simulation import sim_data
from .. import trainer
from .. import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# ---- data settings
name = '/tmp/uci/task01/best'#######dont change this
n_hidden = 128
n_samples = 20
max_iter = 10000
batch_size = 16
L = 10000

# ---- choose the missing model
mprocess = 'linear'

# ---- number of runs
runs = 50
RMSE_model = []

for run in range(runs):
    
    # ---- load data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = np.array(pd.read_csv(url, low_memory=False, sep=';'))#,.; for banknote, wine
    # ---- drop the classification attribute
    data = data[:, :-1]#[:, 2:]
    ###
    data = data.astype(float)
    ###
    # ---- standardize data
    # important*** standardize first
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)
    # ----
    
    # nsc on uci data
    X, R = sim_data.simulate_missing(X = data, missing_type = "general MNAR",censor_linearity = "linear", seed = run) #dimH = n_hidden,#same seed 0 # default : linear censoring#, missing_type = "MCAR":bad performance

    data, Xnan, Xz, _ = sim_data.get_missingdata(X, R) #data is all observed for R not all 0, Xnan: with nan, Xz: replace nan with 0s

    N, D = data.shape
    dl = D - 1
    
    np.random.seed(run)
    # ---- random permutation
    p = np.random.permutation(N) # same dataset, different permutation each run
    data = data[p, :]
    Xnan = Xnan[p, :]
    Xz = Xz[p, :]
# =============================================================================
#     # ---- introduce missing process
#     Xnan, Xz = introduce_mising(data)
# =============================================================================
    S = np.array(~np.isnan(Xnan), dtype=float)
    data_S = np.concatenate((data, S), axis = 1)

# =============================================================================
#     train_id = random.sample(list(range(N)), int(N*0.9))#without replacement.
#     val_id = [id for id in list(range(N)) if id not in train_id]
# =============================================================================

    # training data 
    #Xtrain = data[train_id,:]
    Xtrainnan = Xnan#[train_id,:]
    Xtrainz = Xz#[train_id,:]

    # validation data 
    #Xval = data[val_id,:]
    Xvalnan = Xnan#[val_id,:]
    Xvalz = Xz#[val_id,:]
