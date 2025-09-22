import numpy as np
import pandas as pd
import os
import sys
import random
sys.path.append(os.getcwd())
from ..IM-IWAE import newModel
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
    data = data[:, :-1]
    data = data.astype(float)
    # ---- standardize data
    # important*** standardize first
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)
    
    # mnar on uci data
    X, R = sim_data.simulate_missing(X = data, missing_type = "general MNAR",censor_linearity = "linear", seed = run) 
    data, Xnan, Xz, _ = sim_data.get_missingdata(X, R) #data is all observed for R not all 0, Xnan: with nan, Xz: replace nan with 0s

    N, D = data.shape
    dl = D - 1
    
    np.random.seed(run)
    # ---- random permutation
    p = np.random.permutation(N) # same dataset, different permutation each run
    data = data[p, :]
    Xnan = Xnan[p, :]
    Xz = Xz[p, :]

    S = np.array(~np.isnan(Xnan), dtype=float)
    data_S = np.concatenate((data, S), axis = 1)

    # training data 
    Xtrainnan = Xnan
    Xtrainz = Xz

    # validation data 
    Xvalnan = Xnan
    Xvalz = Xz

    print("fitting new Model")
    # ---------------------- #
    # ---- fit new model---- #
    # ---------------------- #
    model = newModel(Xtrainnan, Xvalnan, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process = 'linear_nsc', name=name)

    # ---- do the training
    trainer.train(model, batch_size=batch_size, max_iter=max_iter, name=name + 'model')

    # ---- find imputation RMSE
    #RMSE_model_ez.append(utils.imputationRMSE(model, Xtrain, Xz, Xnan, S, L)[0])
    RMSE_model.append(utils.not_imputationRMSE(model, data, Xz, Xnan, S, L)[0])
    
    #print(RMSE_model_ez)
    print(RMSE_model)
    print("RMSE_model = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_model), np.std(RMSE_model)))
