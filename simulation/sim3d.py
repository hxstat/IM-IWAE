import numpy as np
import pandas as pd
import os
import sys
import random
sys.path.append(os.getcwd())
import sim_data
from mmd_measure import mmd
from IM_IWAE import newModel
import trainer
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


### encoder add mask?

# ---- data settings
name = '/tmp/sim3d/task01/best'
n_hidden = 128
n_samples = 20
max_iter = 10000
batch_size = 16
L = 10000

# ---- choose the missing model
mprocess = 'linear'
# ---- number of runs

runs = 50
xmmd_model = []
jmmd_model = []
xmmd_model1 = []
jmmd_model1 = []


RMSE_model = []


for run in range(runs):
    
    # ---- load data
    dl = 3
    
    # without latent variables for R:
    X, R = sim_data.simulate_data(dimZ = dl,  dimX = 3, censor_linearity = "linear",addon = 3, seed = run) #censor_linearity = "nonlinear"
    
    # with latent variables for R:
    # X, R = sim_data.simulate_data_zt(dimZ = dl,  dimX = 3, censor_linearity = "linear", addon = 3, seed = run)
    
    data, Xnan, Xz, _ = sim_data.get_missingdata(X, R) #data is all observed for R not all 0, Xnan: with nan, Xz: replace nan with 0s
    
    N, D = data.shape

    
    np.random.seed(run)
    # ---- random permutation
    p = np.random.permutation(N) # same dataset, different permutation each run
    data = data[p, :]
    Xnan = Xnan[p, :]
    Xz = Xz[p, :]
    S = np.array(~np.isnan(Xnan), dtype=float)
    
    data_S = np.concatenate((data, S), axis = 1)

    # training data 
    Xtrainnan = Xnan#[train_id,:]
    Xtrainz = Xz#[train_id,:]

    # validation data 
    Xvalnan = Xnan#[val_id,:]
    Xvalz = Xz#[val_id,:]



    print("fitting new Model")
    # ---------------------- #
    # ---- fit new model---- #
    # ---------------------- #
    model = newModel(Xtrainnan, Xvalnan, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process = 'linear_nsc', name=name) #missing_process = 'nonlinear_nsc_shareparam'

    # ---- do the training
    trainer.train(model, batch_size=batch_size, max_iter=max_iter, name=name + 'model')
    
    RMSE_model.append(utils.not_imputationRMSE(model, data, Xz, Xnan, S, L)[0])
    print(RMSE_model)
    
    #1 sample
    x_gen, s_gen = utils.generate2(model, N, ns = 1)
    x_gen = np.mean(x_gen,axis = 1)
    s_gen = np.mean(s_gen,axis = 1)
    xs_gen = np.concatenate((x_gen, s_gen), axis = 1)
    xmmd = mmd(x_gen, data)
    jmmd = mmd(xs_gen, data_S)
    xmmd_model1.append(xmmd)
    jmmd_model1.append(jmmd)
    print(xmmd_model1)
    print(jmmd_model1)

print("RMSE_model = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_model), np.std(RMSE_model)))
