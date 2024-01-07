import numpy as np
from scipy.stats import uniform

# old
def old_self_censoring(X):
    N, D = X.shape
    Xnan = X.copy()

    # ---- MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz
# ---- useful functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def map_binary(inp): #quick function for calculating empirical probabilities
    n, dim =inp.shape
    tens_arr = np.zeros(n)#.cuda()
    for i in range(dim):
        tens_arr += 2**i*(inp[:,i])
    cnt_arr = np.zeros(2**dim)
    for i in range(2**dim):
        cnt_arr[i] = np.sum(tens_arr == i)
    return cnt_arr, tens_arr

def simulate_complete_data(seed = 0,# missing = True, 
                  dimZ = 10, dimX = 680, dimH = 50, N_sim = 212):
    np.random.seed(seed)
    # ---- generate X from Z (independent X even given Z)
    Z = np.random.normal(size=(N_sim, dimZ))
    X = np.empty((N_sim, dimX))
    
    #for d in range(dimX):
    W1 = np.random.normal(size=(dimZ, dimH))#+d
    b1 = np.random.normal(loc = 0, scale = 0.1, size = (N_sim, dimH))
    W2 = np.random.normal(size=(dimH, dimH))
    b2 = np.random.normal(loc = 0, scale = 0.1, size = (N_sim, dimH))
    W3 = np.random.normal(size=(dimH, dimX))
    epsilon = np.random.normal(loc = 0, scale = 0.1, size = (N_sim, dimX))

    h1 = np.tanh(np.matmul(Z, W1)+b1)#np.concatenate((X[:,:d], Z), axis = 1)
    h2 = np.tanh(np.matmul(h1, W2)+b2)
    X = np.matmul(h2, W3) + epsilon
        #X[:,d] = (fd + epsilon).reshape(-1)

    
    return X

def simulate_data(seed = 0,# missing = True, 
                  dimZ = 3, dimX = 3, dimH = 50, N_sim = 20000, 
                  corr_strength = "strong", block_size = 1, 
                  missing_type = "no self censoring", censor_linearity = "nonlinear", addon = 0):
    
    np.random.seed(seed)
    # ---- generate X from Z (independent X even given Z)
    Z = np.random.normal(size=(N_sim, dimZ))
    X = np.empty((N_sim, dimX))
    
    #for d in range(dimX):=
    W1 = np.random.normal(size=(dimZ, dimH))#+d
    b1 = np.random.normal(loc = 0, scale = 0.1, size = (N_sim, dimH))
    W2 = np.random.normal(size=(dimH, dimH))
    b2 = np.random.normal(loc = 0, scale = 0.1, size = (N_sim, dimH))
    W3 = np.random.normal(size=(dimH, dimX))
    epsilon = np.random.normal(loc = 0, scale = 0.1, size = (N_sim, dimX))

    h1 = np.tanh(np.matmul(Z, W1)+b1)#np.concatenate((X[:,:d], Z), axis = 1)
    h2 = np.tanh(np.matmul(h1, W2)+b2)
    X = np.matmul(h2, W3) + epsilon
        #X[:,d] = (fd + epsilon).reshape(-1)
    X_unscaled = X.copy()
    
    dimR = int(dimX/block_size)

    if corr_strength == "strong":
        u_indep = uniform.rvs(size=(N_sim,1))
        u = np.repeat(u_indep, dimR, axis=1)      
    elif corr_strength == "none":
        u = uniform.rvs(size=(N_sim, dimR))
    
    # ---- generate R based on noise outsourcing lemma
    R_logits = np.empty((N_sim, dimR))
    Rp = np.empty((N_sim, dimR))
    # R = np.zeros((N_sim, dimR), dtype = int)
    
    if missing_type == "no self censoring":
        if censor_linearity == "nonlinear":

            for d in range(dimR):
                WR1 = np.random.normal(size=(dimX-block_size, dimH))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, dimH))
                WR2 = np.random.normal(size=(dimH, 1))
                bR2 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
                hR1 = np.tanh(np.matmul(X_noself, WR1)+bR1)
                R_logits[:,d] = (np.matmul(hR1, WR2)+bR2).reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
        elif censor_linearity == "linear":

            for d in range(dimR):
            
                WR1 = np.random.normal(size=(dimX-block_size, 1))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
                hR1 = np.matmul(X_noself, WR1)+bR1
                R_logits[:,d] = hR1.reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
                
    elif missing_type == "self censoring":
        if censor_linearity == "nonlinear":
            for d in range(dimR):
            
                WR1 = np.random.normal(size=(block_size, dimH))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, dimH))
                WR2 = np.random.normal(size=(dimH, 1))
                bR2 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_self = X[:,d*block_size:(d+1)*block_size]
                hR1 = np.tanh(np.matmul(X_self, WR1)+bR1)
                R_logits[:,d] = (np.matmul(hR1, WR2)+bR2).reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d])# Rpd #Rpd = #+3*d
        elif censor_linearity == "linear":
            for d in range(dimR):
            
                WR1 = np.random.normal(size=(block_size, 1))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
                X_self = X[:,d*block_size:(d+1)*block_size]
                hR1 = np.matmul(X_self, WR1)+bR1
                R_logits[:,d] = hR1.reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d])# Rpd #Rpd = #+3*d
    elif missing_type == "general MNAR":
        if censor_linearity == "nonlinear":
            WR1 = np.random.normal(size=(dimX, dimH))#+d
            bR1 = np.random.normal(loc = 0, scale = 0.1, size = (1, dimH))           
            WR2 = np.random.normal(size=(dimH, dimR))
            bR2 = np.random.normal(loc = 0, scale = 0.1, size = (1, dimR))
            
            hR1 = np.tanh(np.matmul(X, WR1)+bR1)
            R_logits = np.matmul(hR1, WR2)+bR2
            Rp = sigmoid(R_logits)
        elif censor_linearity == "linear":
            WR1 = np.random.normal(size=(dimX, dimR))#+d
            bR1 = np.random.normal(loc = 0, scale = 0.1, size = (1, dimR))           

            R_logits= np.matmul(X, WR1)+bR1
            Rp = sigmoid(R_logits)
    elif missing_type == "MCAR":
        Rp = np.repeat(uniform.rvs(size=(1, dimR)), N_sim, 0)
        
    R = (u < Rp).astype(int)
    
    
    # Standardize data X
    X = X_unscaled - np.mean(X_unscaled, axis=0)
    X = X / np.std(X, axis=0)  
    
    
    if dimR <= 10:
        print("R distribution")
        R_probs = map_binary(R)[0]/N_sim #R.shape[0]
        print(R_probs)
    print("missing rate")
    #print(R==0)
    print(np.sum(R==0)/N_sim/dimR)
    #dimR
    #dimX
    return X, R #, Xorg, Xnan, Xm, Rm

def get_missingdata(X, R):

    dimR = R.shape[1]
    dimX = X.shape[1]
    block_size = int(dimX/dimR)
    R_rep = np.repeat(R, block_size, axis=1)

    obs_id = np.sum(R_rep, axis = 1) > 0
    Rm = R_rep[obs_id, :]
    Xorg = X[obs_id, :]
    Xm = Xorg*Rm
    Xnan = Xm.copy()
    Xnan[Rm == 0] = np.nan
    print("observable missing rate")
    print(np.sum(Rm==0)/Rm.shape[0]/Rm.shape[1])

    return Xorg, Xnan, Xm, Rm


def get_missingdata_wrate(X, R):

    dimR = R.shape[1]
    dimX = X.shape[1]
    block_size = int(dimX/dimR)
    R_rep = np.repeat(R, block_size, axis=1)

    obs_id = np.sum(R_rep, axis = 1) > 0
    Rm = R_rep[obs_id, :]
    Xorg = X[obs_id, :]
    Xm = Xorg*Rm
    Xnan = Xm.copy()
    Xnan[Rm == 0] = np.nan
    
    rate = np.sum(Rm==0)/Rm.shape[0]/Rm.shape[1]
    print("observable missing rate")
    print(rate)

    return Xorg, Xnan, Xm, Rm, rate

def get_allmissingdata(X, R):
    
    dimR = R.shape[1]
    dimX = X.shape[1]
    block_size = int(dimX/dimR)
    R_rep = np.repeat(R, block_size, axis=1)

    #obs_id = np.sum(R_rep, axis = 1) > 0
    Rm = R_rep#[obs_id, :]
    Xorg = X#[obs_id, :]
    Xm = Xorg*Rm
    Xnan = Xm.copy()
    Xnan[Rm == 0] = np.nan
    print("observable missing rate")
    print(np.sum(Rm==0)/Rm.shape[0]/Rm.shape[1])

    return Xorg, Xnan, Xm, Rm

def simulate_bankall_missing(X, seed = 0, dimH = 50,# missing = True, 
                  corr_strength = "strong", block_size = 1, 
                  missing_type = "no self censoring", censor_linearity = "nonlinear", addon = 0):
    np.random.seed(seed)
    
    N_sim, dimX = X.shape

    #X_unscaled = X.copy()
    
    dimR = int(dimX/block_size)

    if corr_strength == "strong":
        u_indep = uniform.rvs(size=(N_sim,1))
        u = np.repeat(u_indep, dimR, axis=1)      
    elif corr_strength == "none":
        u = uniform.rvs(size=(N_sim, dimR))
    
    # ---- generate R based on noise outsourcing lemma
    R_logits = np.empty((N_sim, dimR))
    Rp = np.ones((N_sim, dimR))
    # R = np.zeros((N_sim, dimR), dtype = int)
    
    if missing_type == "no self censoring":
        if censor_linearity == "nonlinear":

            d=2
            WR1 = np.random.normal(size=(dimX-block_size, dimH))
            bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, dimH))
            WR2 = np.random.normal(size=(dimH, 1))
            bR2 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
        
            X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
            hR1 = np.tanh(np.matmul(X_noself, WR1)+bR1)
            R_logits[:,d] = (np.matmul(hR1, WR2)+bR2).reshape(-1)
            Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
            for d in range(9,16):
                WR1 = np.random.normal(size=(dimX-block_size, dimH))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, dimH))
                WR2 = np.random.normal(size=(dimH, 1))
                bR2 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
                hR1 = np.tanh(np.matmul(X_noself, WR1)+bR1)
                R_logits[:,d] = (np.matmul(hR1, WR2)+bR2).reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
        elif censor_linearity == "linear":

            d=2
            
            WR1 = np.random.normal(size=(dimX-block_size, 1))
            bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
        
            X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
            hR1 = np.matmul(X_noself, WR1)+bR1
            R_logits[:,d] = hR1.reshape(-1)
            Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
            for d in range(9,16):
                WR1 = np.random.normal(size=(dimX-block_size, 1))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
                hR1 = np.matmul(X_noself, WR1)+bR1
                R_logits[:,d] = hR1.reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
                
    
        
    R = (u < Rp).astype(int)
    
    
    if dimR <= 10:
        print("R distribution")
        R_probs = map_binary(R)[0]/N_sim #R.shape[0]
        print(R_probs)
    print("missing rate")
    #print(R==0)
    print(np.sum(R==0)/N_sim/dimR)
    #dimR
    #dimX
    return X, R #, Xorg, Xnan, Xm, Rm
def simulate_banknum_missing(X, seed = 0, dimH = 50,# missing = True, 
                  corr_strength = "strong", block_size = 1, 
                  missing_type = "no self censoring", censor_linearity = "nonlinear", addon = 0):
    np.random.seed(seed)
    
    N_sim, dimX = X.shape

    #X_unscaled = X.copy()
    
    dimR = int(dimX/block_size)

    if corr_strength == "strong":
        u_indep = uniform.rvs(size=(N_sim,1))
        u = np.repeat(u_indep, dimR, axis=1)      
    elif corr_strength == "none":
        u = uniform.rvs(size=(N_sim, dimR))
    
    # ---- generate R based on noise outsourcing lemma
    R_logits = np.empty((N_sim, dimR))
    Rp = np.ones((N_sim, dimR))
    # R = np.zeros((N_sim, dimR), dtype = int)
    
    if missing_type == "no self censoring":
        if censor_linearity == "nonlinear":

            d=2
            WR1 = np.random.normal(size=(dimX-block_size, dimH))
            bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, dimH))
            WR2 = np.random.normal(size=(dimH, 1))
            bR2 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
        
            X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
            hR1 = np.tanh(np.matmul(X_noself, WR1)+bR1)
            R_logits[:,d] = (np.matmul(hR1, WR2)+bR2).reshape(-1)
            Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
        elif censor_linearity == "linear":

            d=2
            
            WR1 = np.random.normal(size=(dimX-block_size, 1))
            bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
        
            X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
            hR1 = np.matmul(X_noself, WR1)+bR1
            R_logits[:,d] = hR1.reshape(-1)
            Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
                
    
        
    R = (u < Rp).astype(int)
    
    if dimR <= 10:
        print("R distribution")
        R_probs = map_binary(R)[0]/N_sim #R.shape[0]
        print(R_probs)
    print("missing rate")
    #print(R==0)
    print(np.sum(R==0)/N_sim/dimR)
    #dimR
    #dimX
    return X, R #, Xorg, Xnan, Xm, Rm


def simulate_missing(X, seed = 0, dimH = 50,# missing = True, 
                  corr_strength = "strong", block_size = 1, 
                  missing_type = "no self censoring", censor_linearity = "nonlinear", addon = 0, standardize = False):
    np.random.seed(seed)
    
    N_sim, dimX = X.shape

    X_unscaled = X.copy()
    
    dimR = int(dimX/block_size)

    if corr_strength == "strong":
        u_indep = uniform.rvs(size=(N_sim,1))
        u = np.repeat(u_indep, dimR, axis=1)      
    elif corr_strength == "none":
        u = uniform.rvs(size=(N_sim, dimR))
    
    # ---- generate R based on noise outsourcing lemma
    R_logits = np.empty((N_sim, dimR))
    Rp = np.empty((N_sim, dimR))
    # R = np.zeros((N_sim, dimR), dtype = int)
    
    if missing_type == "no self censoring":
        if censor_linearity == "nonlinear":

            for d in range(dimR):
                WR1 = np.random.normal(size=(dimX-block_size, dimH))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, dimH))
                WR2 = np.random.normal(size=(dimH, 1))
                bR2 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
                hR1 = np.tanh(np.matmul(X_noself, WR1)+bR1)
                R_logits[:,d] = (np.matmul(hR1, WR2)+bR2).reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
        elif censor_linearity == "linear":

            for d in range(dimR):
            
                WR1 = np.random.normal(size=(dimX-block_size, 1))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
                hR1 = np.matmul(X_noself, WR1)+bR1
                R_logits[:,d] = hR1.reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
                
    elif missing_type == "self censoring":
        if censor_linearity == "nonlinear":
            for d in range(dimR):
            
                WR1 = np.random.normal(size=(block_size, dimH))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, dimH))
                WR2 = np.random.normal(size=(dimH, 1))
                bR2 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_self = X[:,d*block_size:(d+1)*block_size]
                hR1 = np.tanh(np.matmul(X_self, WR1)+bR1)
                R_logits[:,d] = (np.matmul(hR1, WR2)+bR2).reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d])# Rpd #Rpd = #+3*d
        elif censor_linearity == "linear":
            for d in range(dimR):
            
                WR1 = np.random.normal(size=(block_size, 1))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
                X_self = X[:,d*block_size:(d+1)*block_size]
                hR1 = np.matmul(X_self, WR1)+bR1
                R_logits[:,d] = hR1.reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d])# Rpd #Rpd = #+3*d
    elif missing_type == "general MNAR":
        if censor_linearity == "nonlinear":
            WR1 = np.random.normal(size=(dimX, dimH))#+d
            bR1 = np.random.normal(loc = 0, scale = 0.1, size = (1, dimH))           
            WR2 = np.random.normal(size=(dimH, dimR))
            bR2 = np.random.normal(loc = 0, scale = 0.1, size = (1, dimR))
            
            hR1 = np.tanh(np.matmul(X, WR1)+bR1)
            R_logits = np.matmul(hR1, WR2)+bR2
            Rp = sigmoid(R_logits+addon)
        elif censor_linearity == "linear":
            WR1 = np.random.normal(size=(dimX, dimR))#+d
            bR1 = np.random.normal(loc = 0, scale = 0.1, size = (1, dimR))           

            R_logits= np.matmul(X, WR1)+bR1
            Rp = sigmoid(R_logits+addon)
    elif missing_type == "MCAR":
        Rp = np.repeat(uniform.rvs(size=(1, dimR)), N_sim, 0)
        
    R = (u < Rp).astype(int)
    
    
    # Standardize data X
    if standardize:
        X = X_unscaled - np.mean(X_unscaled, axis=0)
        X = X / np.std(X, axis=0)  
    
    
    if dimR <= 10:
        print("R distribution")
        R_probs = map_binary(R)[0]/N_sim #R.shape[0]
        print(R_probs)
    print("missing rate")
    #print(R==0)
    print(np.sum(R==0)/N_sim/dimR)

    return X, R #, Xorg, Xnan, Xm, Rm


def simulate_blockmissing(X, seed = 0, dimH = 50,# missing = True, 
                  corr_strength = "strong", block_shape = [267,113,300], 
                  missing_type = "no self censoring", censor_linearity = "nonlinear", addon = 200):
    np.random.seed(seed)
    
    N_sim, dimX = X.shape

    # X_unscaled = X.copy()
    
    dimR = len(block_shape)

    if corr_strength == "strong":
        u_indep = uniform.rvs(size=(N_sim,1))
        u = np.repeat(u_indep, dimR, axis=1)      
    elif corr_strength == "none":
        u = uniform.rvs(size=(N_sim, dimR))
    
    # ---- generate R based on noise outsourcing lemma
    R_logits = np.empty((N_sim, dimR))
    Rp = np.empty((N_sim, dimR))
    # R = np.zeros((N_sim, dimR), dtype = int)
    
    if missing_type == "no self censoring":
        if censor_linearity == "nonlinear":
            begin = 0
            end = block_shape[0]
            for d in range(dimR):
                WR1 = np.random.normal(size=(dimX-block_shape[d], dimH))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, dimH))
                WR2 = np.random.normal(size=(dimH, 1))
                bR2 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:begin], X[:,end:]), axis = 1)
                if d < dimR-1:
                    begin = end
                    end = end + block_shape[d+1]
                hR1 = np.tanh(np.matmul(X_noself, WR1)+bR1)
                R_logits[:,d] = (np.matmul(hR1, WR2)+bR2).reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
        elif censor_linearity =="linear":
            begin = 0
            end = block_shape[0]
            for d in range(dimR):
                WR1 = np.random.normal(size=(dimX-block_shape[d], 1))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:begin], X[:,end:]), axis = 1)
                if d < dimR-1:
                    begin = end
                    end = end + block_shape[d+1]
                hR1 = np.matmul(X_noself, WR1)+bR1
                R_logits[:,d] = hR1.reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
        
    R = (u < Rp).astype(int)
    
    
    R = np.repeat(R, block_shape, axis=1)
    print(R.shape)

    print("missing rate")
    #print(R==0)
    print(np.sum(R==0)/N_sim/np.sum(block_shape))
    #dimR
    #dimX
    
    return X, R #, Xorg, Xnan, Xm, Rm



def simulate_data_zt(seed = 0,# missing = True, 
                  dimZ = 3, dimZt = 1, dimX = 3, dimH = 50, N_sim = 20000, 
                  corr_strength = "strong", block_size = 1, 
                  missing_type = "no self censoring", censor_linearity = "nonlinear", addon = 0):
    
    np.random.seed(seed)
    # ---- generate X from Z (independent X even given Z)
    Z = np.random.normal(size=(N_sim, dimZ))
    Zt = np.random.normal(size=(N_sim, dimZt))
    X = np.empty((N_sim, dimX))
    
    #for d in range(dimX):=
    W1 = np.random.normal(size=(dimZ, dimH))#+d
    b1 = np.random.normal(loc = 0, scale = 0.1, size = (N_sim, dimH))
    W2 = np.random.normal(size=(dimH, dimH))
    b2 = np.random.normal(loc = 0, scale = 0.1, size = (N_sim, dimH))
    W3 = np.random.normal(size=(dimH, dimX))
    epsilon = np.random.normal(loc = 0, scale = 0.1, size = (N_sim, dimX))

    h1 = np.tanh(np.matmul(Z, W1)+b1)#np.concatenate((X[:,:d], Z), axis = 1)
    h2 = np.tanh(np.matmul(h1, W2)+b2)
    X = np.matmul(h2, W3) + epsilon
        #X[:,d] = (fd + epsilon).reshape(-1)
    X_unscaled = X.copy()
    
    dimR = int(dimX/block_size)

    u = uniform.rvs(size=(N_sim, dimR))
    
    # ---- generate R from ztilde
    R_logits = np.empty((N_sim, dimR))
    Rp = np.empty((N_sim, dimR))
    # R = np.zeros((N_sim, dimR), dtype = int)
    
    if missing_type == "no self censoring":
        if censor_linearity == "nonlinear":

            for d in range(dimR):
                WR1 = np.random.normal(size=(dimX-block_size+dimZt, dimH))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, dimH))
                WR2 = np.random.normal(size=(dimH, 1))
                bR2 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
                hR1 = np.tanh(np.matmul(np.concatenate((X_noself,Zt), axis = 1), WR1)+bR1)
                R_logits[:,d] = (np.matmul(hR1, WR2)+bR2).reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
        elif censor_linearity == "linear":

            for d in range(dimR):
            
                WR1 = np.random.normal(size=(dimX-block_size+dimZt, 1))
                bR1 = np.random.normal(loc = 0, scale = 0.1, size=(1, 1))
            
                X_noself = np.concatenate((X[:,:d*block_size], X[:,(d+1)*block_size:]), axis = 1)
                hR1 = np.matmul(np.concatenate((X_noself,Zt), axis = 1), WR1)+bR1
                R_logits[:,d] = hR1.reshape(-1)
                Rp[:,d] = sigmoid(R_logits[:,d]+addon)# Rpd #Rpd = #+3*d
                

        
    R = (u < Rp).astype(int)
    
    
    # Standardize data X

    X = X_unscaled - np.mean(X_unscaled, axis=0)
    X = X / np.std(X, axis=0)  
    
    
    if dimR <= 10:
        print("R distribution")
        R_probs = map_binary(R)[0]/N_sim #R.shape[0]
        print(R_probs)
    print("missing rate")
    #print(R==0)
    print(np.sum(R==0)/N_sim/dimR)
    #dimR
    #dimX
    return X, R #, Xorg, Xnan, Xm, Rm

