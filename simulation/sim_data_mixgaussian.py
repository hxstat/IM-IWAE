# -*- coding: utf-8 -*-

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_probability as tfp
import keras
import pandas as pd
import time

from scipy.stats import bernoulli

# simulate Gaussian data
def sim_gaussian(N_sim = 5000, seed = 0):
    
    np.random.seed(seed)
    
    Sigma0 = np.matrix([[4.4, 1.3, -2.8], 
                      [1.3, 3.2, 1.3], 
                      [-2.8, 1.3,3.5]])
    h_r = np.matrix([[1.4, 1.6, 0.9], 
                    [1.9, 1.1, 1.4],
                    [1.9, 1.6, 0.2], 
                    [0.5, 1.9, 2.1], 
                    [0.5, 2.4, 0.9],
                    [1.0, 1.9, 1.4],
                    [1.0, 2.4, 0.2],
                    [1.4, 1.1, 2.1]])
    mu_r = np.transpose(np.matmul(Sigma0, np.transpose(h_r)))
    r_order = np.array([[1,0,0],
                        [0,1,0],
                        [1,1,0],
                        [0,0,1],
                        [1,0,1],
                        [0,1,1],
                        [1,1,1],
                        [0,0,0]])
    p_r =[0.169, 0.153,0.136,0.119,0.102,0.085,0.169,0.068]
    
    sim_cat = np.random.multinomial(N_sim, p_r)
    mask_mat = np.zeros((N_sim, 3),dtype=np.float32)
    X = np.empty((N_sim, 3))
    cnt = 0
    for i in range(len(sim_cat)):
        this_cat_size = sim_cat[i]
        this_cat_mask_single = r_order[i,:]
        this_cat_mask = np.tile(this_cat_mask_single, (this_cat_size, 1))
        mask_mat[cnt:cnt+this_cat_size,:] = this_cat_mask     
        this_cat_X = np.random.multivariate_normal(np.ravel(mu_r[i,:]), Sigma0, this_cat_size)
        X[cnt:(cnt+this_cat_size),:] = this_cat_X
        cnt = cnt + this_cat_size
        
    return X, mask_mat
