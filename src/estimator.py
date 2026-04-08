import numpy as np

def DifferenceInMeans(Y1, Y0, Z):
    """Computes the difference in means vector tau_hat"""
    tauhat = Y1[Z==1].mean(axis=0) - Y0[Z==0].mean(axis=0)
    return tauhat
