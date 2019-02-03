import numpy as np
import pandas as pd
import math
import streamlit as st

def oversample(X_train, y_train, z_train, target_col, bias_names):

    # Calculate some stats on out training set so we can augment
    n_train = X_train.shape[0]
    n_z = [0]*2
    i_t_z = [None]*2
    n_t_z = [0]*2
    f_t_z = [0]*2
    for z in [0,1]:
        n_z[z] = X_train[z_train==z].shape[0]
        i_t_z[z] = X_train.index[np.logical_and(z_train==z, y_train==1)].tolist()
        n_t_z[z] = len(i_t_z[z])
        f_t_z[z] = n_t_z[z]/n_z[z]
        st.write("Proportion of {} for which {} before oversampling:  {:2.2%}".format(bias_names[z], target_col, f_t_z[z]))

    # Augment the training set by oversampling rich women
    frac, integer = math.modf(f_t_z[1] / f_t_z[0])
    n_frac = int(frac*n_t_z[0])
    i_frac = np.random.choice(i_t_z[0], n_frac)

    X_new = X_train.copy()
    y_new = y_train.copy()
    z_new = z_train.copy()
    for i in range(int(integer)):
        X_new = pd.concat([X_new, X_train.loc[i_t_z[0]]], ignore_index=True)
        y_new = pd.concat([y_new, y_train.loc[i_t_z[0]]], ignore_index=True)
        z_new = pd.concat([z_new, z_train.loc[i_t_z[0]]], ignore_index=True)
    X_new = pd.concat([X_new, X_train.loc[i_frac]], ignore_index=True)
    y_new = pd.concat([y_new, y_train.loc[i_frac]], ignore_index=True)
    z_new = pd.concat([z_new, z_train.loc[i_frac]], ignore_index=True)

    # Check we have the right number of data points in our augmented training set
    #st.write('Should have ',n_train+integer*n_t_z[0]+n_frac,' data points:', X_new.shape[0])
    return X_new, y_new, z_new
