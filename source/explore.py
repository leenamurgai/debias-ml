import pandas as pd
import streamlit as st
from plots import target_by_bias_table_histogram

def basic_stats(data_df, bias_cols, target_col, pos_target):

    st.write('')
    st.write('**Some basic statistics:**')
    st.write('Number of data points =', len(data_df.index))
    st.write('Number of features =', len(data_df.columns)-1)

    target_names = tuple(data_df[target_col].unique())
    if target_names[0]==pos_target:
        target_names = target_names[1], target_names[0]
    bias_names = [None]*len(bias_cols)
    for i, col in enumerate(bias_cols):
        bias_names[i] = tuple(data_df[col].unique())

    n_target = [0]*len(target_names)
    for i, t in enumerate(target_names):
        n_target[i] = data_df[data_df[target_col] == t].shape[0]
        st.write('Number of', target_col, t, '= ', n_target[i])

    n_bias = [None]*len(bias_cols)
    for j, col in enumerate(bias_cols):
        n_bias[j] = [0]*len(bias_names[j])
        for i, b in enumerate(bias_names[j]):
            n_bias[j][i] = data_df[data_df[col] == b].shape[0]
            st.write('Number of {} {} ='.format(col, b), n_bias[j][i])

    pos_bias_labels = {}
    columns = list(data_df)
    for k, col in enumerate(bias_cols):
        pos_bias_labels[col] = []
        target_by_bias = data_df.groupby([col,target_col]).count()[columns[0]]
        temp = [None]*len(target_names)
        for i, t in enumerate(target_names):
            temp[i] = [None]*len(bias_names[k])
            for j, b in enumerate(bias_names[k]):
                temp[i][j] = target_by_bias[b, t]
        target_by_bias_df = pd.DataFrame(data = temp, index = target_names, columns=bias_names[k])
        target_by_bias_df = target_by_bias_df/target_by_bias_df.sum()

        # figure out how to split the bias categories so there are only 2 and ultimately 1 feature to train on
        mean_prop = target_by_bias_df.mean(axis=1)
        for b in list(target_by_bias_df):
            if target_by_bias_df[b].loc[pos_target]>mean_prop[pos_target]:
                pos_bias_labels[col].append(b)

        target_by_bias_table_histogram(target_by_bias_df, target_col, col, 'original-data-'+col)
        
    return pos_bias_labels


def top_n_correlated_features(data_df, sensitive_feature, n):
    corr_df = data_df.corr()
    sex_corrs = corr_df.reindex(corr_df[sensitive_feature].abs().sort_values(ascending=False).index)[sensitive_feature]
    return sex_corrs.iloc[:n]
