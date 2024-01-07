import numpy as np
import pandas as pd
import os
import sys
import random
sys.path.append(os.getcwd())
import sim_data
import sim_data_mixgaussian
from ..NIM-IWAE import newModel
import ..trainer
import ..utils
from mmd_measure import mmd
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


# ---- data settings
name = '/tmp/mixgauss/task01/best'
n_hidden = 128
n_samples = 20
max_iter = 20000
batch_size = 16
L = 10000

# ---- choose the missing model
mprocess = 'linear'
# ---- number of runs
runs = 100
mean_l3_model = []
mean_l3_model_gen = []
RMSE_model = []



for run in range(runs):

    # 3D mixture Gaussian 
    dl = 3
    X, R = sim_data_mixgaussian.sim_gaussian(seed = run)

    data, Xnan, Xz, _ = sim_data.get_missingdata(X, R)
    #data, Xnan, Xz, _ = sim_data.get_full_missingdata(X, R) #data is all observed for R with all 0, Xnan: with nan, Xz: replace nan with 0s
    
    N, D = data.shape

    
    np.random.seed(run)
    # ---- random permutation
    p = np.random.permutation(N) # same dataset, different permutation each run
    data = data[p, :]
    Xnan = Xnan[p, :]
    Xz = Xz[p, :]
    S = np.array(~np.isnan(Xnan), dtype=float)
    

    # for generative models; train validation split
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
    model = newModel(Xtrainnan, Xvalnan, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process = 'linear_nsc', name=name, complexity = "med")

    # ---- do the training
    trainer.train(model, batch_size=batch_size, max_iter=max_iter, name=name + 'model')

    # ---- find imputation RMSE
    imp_res = utils.not_imputationRMSE(model, data, Xz, Xnan, S, L)
    RMSE_model.append(imp_res[0])

    print(RMSE_model)
    
    m3 = np.mean(imp_res[1][:,2])
    mean_l3_model.append(m3)    
    print(m3)
    

    #1 sample
    x_gen, s_gen = utils.generate2(model, N, ns = 1)
    x_gen = np.mean(x_gen,axis = 1)
    s_gen = np.mean(s_gen,axis = 1)
    xs_gen = np.concatenate((x_gen, s_gen), axis = 1)
    
    m3 = np.mean(x_gen[:,2])
    mean_l3_model_gen.append(m3)    
    print(m3)

print("RMSE_model = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_model), np.std(RMSE_model)))
print("mean_l3_model = {0:.5f} +- {1:.5f}".format(np.mean(mean_l3_model), np.std(mean_l3_model)))
print("mean_l3_model_gen = {0:.5f} +- {1:.5f}".format(np.mean(mean_l3_model_gen), np.std(mean_l3_model_gen)))
