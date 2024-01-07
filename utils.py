import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_probability as tfp

def imputationRMSE(model, Xorg, Xz, X, S, L):
    """
    Imputation error of missing data
    """
    N = len(X)

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]

    def imp(model, xz, s, L):
        l_out, log_p_x_given_z, log_p_z, log_q_z_given_x = model.sess.run(
            [model.l_out_mu, model.log_p_x_given_z, model.log_p_z, model.log_q_z_given_x],
            {model.x_pl: xz, model.s_pl: s, model.n_pl: L})

        wl = softmax(log_p_x_given_z + log_p_z - log_q_z_given_x)

        xm = np.sum((l_out.T * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)

        return l_out, wl, xm, xmix

    XM = np.zeros_like(Xorg)
    XMix = np.zeros_like(Xorg)

    for i in range(N):

        xz = Xz[i, :][None, :]
        s = S[i, :][None, :]

        l_out, wl, xm, xmix = imp(model, xz, s, L)

        XM[i, :] = xm
        XMix[i, :] = xmix

        if i % 1000 == 0:
            print('{0} / {1}'.format(i, N))

    return np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S)), XMix


def not_imputationRMSE(model, Xorg, Xz, X, S, L):
    """
    Imputation error of missing data, using the not-MIWAE
    """
    N = len(X)#X is useless

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]

    def imp(model, xz, s, L):
        l_out, log_p_x_given_z, log_p_z, log_q_z_given_x, log_p_s_given_x  = model.sess.run(
            [model.l_out_mu, model.log_p_x_given_z, model.log_p_z, model.log_q_z_given_x, model.log_p_s_given_x],
            {model.x_pl: xz, model.s_pl: s, model.n_pl: L})

        wl = softmax(log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x)

        xm = np.sum((l_out.T * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)

        return l_out, wl, xm, xmix

    XM = np.zeros_like(Xorg)
    XMix = np.zeros_like(Xorg)

    for i in range(N):

        xz = Xz[i, :][None, :]
        s = S[i, :][None, :]

        l_out, wl, xm, xmix = imp(model, xz, s, L)

        XM[i, :] = xm
        XMix[i, :] = xmix

        if i % 1000 == 0:
            print('{0} / {1}'.format(i, N))

    return np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S)), XMix

def not_imputationRMSE_y(model, Xorg, Xz, X, S, L, y):
    """
    Imputation error of missing data, using the not-MIWAE
    """
    N = len(X)

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]

    def imp(model, xz, s, L):
        l_out, log_p_x_given_z, log_p_z, log_q_z_given_x, log_p_s_given_x  = model.sess.run(
            [model.l_out_mu, model.log_p_x_given_z, model.log_p_z, model.log_q_z_given_x, model.log_p_s_given_x],
            {model.x_pl: xz, model.s_pl: s, model.y_pl: y, model.n_pl: L})

        wl = softmax(log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x)

        xm = np.sum((l_out.T * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)

        return l_out, wl, xm, xmix

    XM = np.zeros_like(Xorg)
    XMix = np.zeros_like(Xorg)

    for i in range(N):

        xz = Xz[i, :][None, :]
        s = S[i, :][None, :]

        l_out, wl, xm, xmix = imp(model, xz, s, L)

        XM[i, :] = xm
        XMix[i, :] = xmix

        if i % 1000 == 0:
            print('{0} / {1}'.format(i, N))

    return np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S)), XMix


def generate(model, N_gen, ns = None):
    #generate N(0,1) for latent z
    if ns is None:
        z = np.random.normal(size=(N_gen, model.n_samples, model.n_latent))#N_gen*model.n_samples*model.n_latent)
    else:
        z = np.random.normal(size=(N_gen, ns, model.n_latent))
    x_hat, s_hat = model.generator(z)
    return x_hat, s_hat

def generate2(model, N_gen, ns = None):
    #generate N(0,1) for latent z
    if ns is None:
        z = np.random.normal(size=(N_gen, model.n_samples, model.n_latent+model.n_latent_tilda))
    else:
        z = np.random.normal(size=(N_gen, ns, model.n_latent+model.n_latent_tilda))
    x_hat, s_hat = model.generator(z)
    return x_hat, s_hat
