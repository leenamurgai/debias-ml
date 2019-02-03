import pandas as pd
import streamlit as st
from plots import target_by_bias_table_histogram

def basic_stats(data_df, bias_col, target_col, pos_target):

    st.write('')
    st.write('**Some basic statistics:**')
    st.write('Number of data points =', len(data_df.index))
    st.write('Number of features =', len(data_df.columns)-1)

    target_names = tuple(data_df[target_col].unique())
    if target_names[0]==pos_target:
        target_names = target_names[1], target_names[0]
    bias_names = tuple(data_df[bias_col].unique())

    n_target = [0]*len(target_names)
    for i, t in enumerate(target_names):
        n_target[i] = data_df[data_df[target_col] == t].shape[0]
        st.write('Number of', target_col, t, '= ', n_target[i])
    n_bias = [0]*len(bias_names)
    for i, b in enumerate(bias_names):
        n_bias[i] = data_df[data_df[bias_col] == b].shape[0]
        st.write('Number of {} {} ='.format(bias_col, b), n_bias[i])

    columns = list(data_df)
    target_by_bias = data_df.groupby([bias_col, target_col]).count()[columns[0]]

    temp = [[0,0],[0,0]]
    for i, t in enumerate(target_names):
        for j, b in enumerate(bias_names):
            temp[i][j] = target_by_bias[b, t]
    target_by_bias_df = pd.DataFrame(data = temp, index = target_names, columns=bias_names)
    target_by_bias_table_histogram(target_by_bias_df, target_col, 'original-data')


def top_n_correlated_features(data_df, sensitive_feature, n):
    corr_df = data_df.corr()
    sex_corrs = corr_df.reindex(corr_df[sensitive_feature].abs().sort_values(ascending=False).index)[sensitive_feature]
    return sex_corrs.iloc[:n]
