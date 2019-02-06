import numpy as np
import pandas as pd
import math
import streamlit as st

def oversample(X_train, y_train, Z_train, target_col, bias_cols, bias_col_types):

    # Calculate some stats on out training set so we can augment
    n_train = X_train.shape[0]
    n_biases = Z_train.shape[1]
    n_Z = [[0,0],[0,0]]                    # number of indices corresponding to each group in Z
    i_t_Z = [[None, None], [None, None]]   # indicies corresponding to Z with positive target label
    n_t_Z = [[0,0],[0,0]]                  # number of indices corresponding to Z with positive target
    f_t_Z = [[0,0],[0,0]]                  # faction of indices corresponding to Z with positive target
    for i in [0,1]:
        for j in [0,1]:
            n_Z[i][j] = X_train[ np.logical_and( Z_train[bias_cols[0]]==i , Z_train[bias_cols[1]]==j ) ].shape[0]
            i_t_Z[i][j] = X_train.index[np.logical_and(np.logical_and( Z_train[bias_cols[0]]==i, Z_train[bias_cols[1]]==j ), y_train==1)].tolist()
            n_t_Z[i][j] = len(i_t_Z[i][j])
            f_t_Z[i][j] = n_t_Z[i][j]/n_Z[i][j]

    st.write('**Some stats on the data before oversampling**')
    st.write('Number of data points:', n_train)
    df = pd.DataFrame(data=n_Z, index=bias_col_types[0], columns=bias_col_types[1])
    st.write(df)
    st.write('Number of data points with positive target:')
    df = pd.DataFrame(data=n_t_Z, index=bias_col_types[0], columns=bias_col_types[1])
    st.write(df)
    st.write('Proportion for which:', target_col)
    df = pd.DataFrame(data=f_t_Z, index=bias_col_types[0], columns=bias_col_types[1])
    st.write(df)
    st.write('Bias factors before oversampling:')
    df = f_t_Z[1][1]/df
    st.write(df)

    # Augment the training set by oversampling under-represented classes
    X_new = X_train.copy()
    y_new = y_train.copy()
    Z_new = Z_train.copy()
    for i in [0,1]:
        for j in [0,1]:
            num_new_points = (f_t_Z[1][1]*n_Z[i][j] - n_t_Z[i][j])/(1-f_t_Z[1][1])
            #st.write(i, j, num_new_points)
            if i==0 or j==0:
                #frac, integer = math.modf(f_t_Z[1][1] / f_t_Z[i][j])
                frac, integer = math.modf(num_new_points / n_t_Z[i][j])
                n_frac = int(frac*n_t_Z[i][j])
                i_frac = np.random.choice(i_t_Z[i][j], n_frac)
                for k in range(int(integer)):
                    X_new = pd.concat([X_new, X_train.loc[i_t_Z[i][j]]], ignore_index=True)
                    y_new = pd.concat([y_new, y_train.loc[i_t_Z[i][j]]], ignore_index=True)
                    Z_new = pd.concat([Z_new, Z_train.loc[i_t_Z[i][j]]], ignore_index=True)
                X_new = pd.concat([X_new, X_train.loc[i_frac]], ignore_index=True)
                y_new = pd.concat([y_new, y_train.loc[i_frac]], ignore_index=True)
                Z_new = pd.concat([Z_new, Z_train.loc[i_frac]], ignore_index=True)

    return X_new, y_new, Z_new
