import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from sklearn import metrics

def mmd(X, Y, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


# X given R matching
def compareXgR(R, X, r_sample, x_sample): #quick function for calculating empirical probabilities
    n, dim = R.shape
    m = r_sample.shape[0]
    ncat = 2**dim
    
    tens_arr_R = np.zeros(n)#.cuda()
    for i in range(dim):
        tens_arr_R += 2**i*(R[:,i])
    tens_arr_r = np.zeros(m)#.cuda()
    for i in range(dim):
        tens_arr_r += 2**i*(r_sample[:,i])
        
    cond_mmr = np.zeros(ncat)
    for i in range(ncat): # compare 8 categories of X|R dist.
        R_idx = tens_arr_R == i
        r_idx = tens_arr_r == i
        
        XgR = X[R_idx, :]
        print(XgR.shape)
        xgr = x_sample[r_idx, :]
        print(xgr.shape)
        if xgr.shape[0] == 0:
            cond_mmr[i] = tf.tensor(float('inf'))
        else:
            cond_mmr[i] = mmd(tf.tensor(XgR.tolist()), tf.tensor(xgr.tolist()))
    return cond_mmr
