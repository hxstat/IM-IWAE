import numpy as np
import pandas as pd
import os
import sys
import random
sys.path.append(os.getcwd())
from ..IM-IWAE import newModel
from ..simluation import sim_data
from ..simluation import sim_data_mixgaussian
from .. import trainer
from .. import utils
from ..mmd_measure import mmd
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json



# ---- data settings
name = '/tmp/bin3d/task01/best'
n_hidden = 128
n_samples = 20
max_iter = 10000
batch_size = 16
L = 10000

# ---- choose the missing model
mprocess ='nonlinear_nsc_moreparam'
comp ="high"
# ---- number of runs
runs = 50
#RMSE_model_ez = []



jd_model = []
jd_gina = []
jd_notmiwae = []
jd_notmiwae_linear = []
jd_mean = []
jd_mice = []
jd_RF = []

jd_model_gen = []
jd_gina_gen = []

jd_notmiwae_gen = []
jd_notmiwae_linear_gen = []


for run in range(runs):
    


    dl = 2
    Xnan = pd.read_csv("binary3ddata.csv").to_numpy()
    N, D = Xnan.shape

    
    np.random.seed(run)
    # ---- random permutation
    p = np.random.permutation(N) # same dataset, different permutation each run

    Xnan = Xnan[p, :]
    
    S = np.array(~np.isnan(Xnan), dtype=float)
    Xz = np.nan_to_num(Xnan, nan=0)
    
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
    model = newModel(Xtrainnan, Xvalnan, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process = mprocess, name=name, 
                     complexity = comp, out_dist = "bern", encode_mask = True)

    # ---- do the training
    trainer.train(model, batch_size=batch_size, max_iter=max_iter, name=name + 'model')

    # ---- find imputation RMSE
    
    imp_res = utils.not_imputationRMSE(model, Xz, Xz, Xnan, S, L)
    Xrec = imp_res[1]
    Xrec[Xrec<0.5] = 0
    Xrec[Xrec>=0.5] = 1
    jd = sim_data.map_binary(Xrec)[0]/Xrec.shape[0]
    #jd_model.append(jd)
    #print(jd_model)
    



    jd_imp = {"(0,0,0)":jd[0], "(0,0,1)":jd[1], "(0,1,0)":jd[2], "(0,1,1)":jd[3], "(1,0,0)":jd[4],"(1,0,1)":jd[5], "(1,1,0)":jd[6], "(1,1,1)":jd[7]}
    jd_imp_df = pd.DataFrame(jd_imp, index = [0])
    if run == 0:
        jd_imp_df.to_csv('bin3djd_imp_'+mprocess+'_'+comp+'_dl'+str(dl)+'_epoch'+str(max_iter)+'.csv', index=False)
    else:
        jd_imp_df.to_csv('bin3djd_imp_'+mprocess+'_'+comp+'_dl'+str(dl)+'_epoch'+str(max_iter)+'.csv', mode = 'a', index=False, header = False)


     #1 sample
    x_gen, s_gen = utils.generate2(model, N, ns = 1)
    Xrec = np.mean(x_gen,axis = 1)
    Xrec[Xrec<0.5] = 0
    Xrec[Xrec>=0.5] = 1
    jd_gen = sim_data.map_binary(Xrec)[0]/Xrec.shape[0]


    jd_gen_dict = {"(0,0,0)":jd_gen[0], "(0,0,1)":jd_gen[1], "(0,1,0)":jd_gen[2], "(0,1,1)":jd_gen[3], "(1,0,0)":jd_gen[4],"(1,0,1)":jd_gen[5], "(1,1,0)":jd_gen[6], "(1,1,1)":jd_gen[7]}
    jd_gen_df = pd.DataFrame(jd_gen_dict, index = [0])

