# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import sys
import random
sys.path.append(os.getcwd())
from IM_IWAE import newModel
import sim_data
import trainer
import utils
from mmd_measure import mmd
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# ---- data settings
name = '/tmp/uci/task01/best'
n_hidden = 128
n_samples = 20
max_iter = 20000
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
RMSE_mean = []


for run in range(runs):
    

    dl = 20

    X, R = sim_data.simulate_data(dimZ = 10,  dimX = 30, block_size = 5, censor_linearity = "linear", addon = 3, seed = run)

    data, Xnan, Xz, _ = sim_data.get_missingdata(X, R)
    
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
    Xtrainnan = Xnan
    Xtrainz = Xz

    # validation data 
    Xvalnan = Xnan
    Xvalz = Xz

    print("fitting new Model")
    # ---------------------- #
    # ---- fit new model---- #
    # ---------------------- #
    model = newModel(Xtrainnan, Xvalnan, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process = 'linear_nsc',  name=name, blocks = [5,5,5,5,5,5])  

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
    
# dictionary of lists 
res_dict = {
"rmse_model":RMSE_model,
"xmmd_model1":xmmd_model1,
"jmmd_model1":jmmd_model1
} 

df = pd.DataFrame(res_dict)

print("RMSE_model = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_model), np.std(RMSE_model)))
print("xmmd_model1 = {0:.5f} +- {1:.5f}".format(np.mean(xmmd_model1), np.std(xmmd_model1)))
print("jmmd_model1 = {0:.5f} +- {1:.5f}".format(np.mean(jmmd_model1), np.std(jmmd_model1)))
