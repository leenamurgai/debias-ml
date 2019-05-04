import numpy as np
import pandas as pd
import math
import streamlit as st


class Oversampler:
    def __init__(self, X, y, Z, target_col, bias_cols, bias_col_types):
        self.X = X
        self.y = y
        self.Z = Z
        self.target_col = target_col
        self.bias_cols = bias_cols
        self.bias_col_types = bias_col_types
        self.n_Z   = [[0,0],[0,0]]                  # number of indices corresponding to each group in Z
        self.i_t_Z = [[None, None], [None, None]]   # indicies corresponding to Z with positive target label
        self.n_t_Z = [[0,0],[0,0]]                  # number of indices corresponding to Z with positive target
        self.f_t_Z = [[0,0],[0,0]]                  # proportion of points with positive target labels
        for i in [0,1]:
            for j in [0,1]:
                self.n_Z[i][j] = X[ np.logical_and( Z[bias_cols[0]]==i,
                                                    Z[bias_cols[1]]==j ) ].shape[0]
                self.i_t_Z[i][j] = X.index[np.logical_and(np.logical_and( Z[bias_cols[0]]==i,
                                                                          Z[bias_cols[1]]==j ),
                                                          y==1)].tolist()
                self.n_t_Z[i][j] = len(self.i_t_Z[i][j])
                self.f_t_Z[i][j] = self.n_t_Z[i][j] / self.n_Z[i][j]


    def original_data_stats(self):
        st.write('**Some stats on the original data (before oversampling)**')
        st.write('Number of data points:', self.X.shape[0])
        df = pd.DataFrame(data=self.n_Z,
                          index=self.bias_col_types[0],
                          columns=self.bias_col_types[1])
        st.write(df)
        st.write('Number of data points with', self.target_col)
        df = pd.DataFrame(data=self.n_t_Z,
                          index=self.bias_col_types[0],
                          columns=self.bias_col_types[1])
        st.write(df)
        st.write('Proportion for which:', self.target_col)
        df = pd.DataFrame(data=self.f_t_Z,
                          index=self.bias_col_types[0],
                          columns=self.bias_col_types[1])
        st.write(df)
        st.write('Bias factors before oversampling:')
        df = self.f_t_Z[1][1] / df
        st.write(df)


    def get_oversampled_data(self, oversample_factor=1):
        # Augment the training set by oversampling under-represented classes
        X_new = self.X.copy()
        y_new = self.y.copy()
        Z_new = self.Z.copy()
        for i in [0,1]:
            for j in [0,1]:
                num_new_points = float(oversample_factor)*(self.f_t_Z[1][1]*self.n_Z[i][j] - self.n_t_Z[i][j])/(1.0-self.f_t_Z[1][1])
                #st.write(i, j, num_new_points)
                if i==0 or j==0:
                    #frac, integer = math.modf(f_t_Z[1][1] / f_t_Z[i][j])
                    frac, integer = math.modf(num_new_points / self.n_t_Z[i][j])
                    n_frac = int(frac*self.n_t_Z[i][j])
                    #i_frac = np.random.choice(self.i_t_Z[i][j], n_frac)
                    i_frac = self.i_t_Z[i][j][: n_frac]
                    for k in range(int(integer)):
                        X_new = pd.concat([X_new, self.X.loc[self.i_t_Z[i][j]]], ignore_index=True)
                        y_new = pd.concat([y_new, self.y.loc[self.i_t_Z[i][j]]], ignore_index=True)
                        Z_new = pd.concat([Z_new, self.Z.loc[self.i_t_Z[i][j]]], ignore_index=True)
                    X_new = pd.concat([X_new, self.X.loc[i_frac]], ignore_index=True)
                    y_new = pd.concat([y_new, self.y.loc[i_frac]], ignore_index=True)
                    Z_new = pd.concat([Z_new, self.Z.loc[i_frac]], ignore_index=True)
        return X_new, y_new, Z_new
