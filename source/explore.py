def basic_stats(data_df, bias_col, target_col, pos_target):
    print('Number of data points =', len(data_df.index))
    print('Number of features =', len(data_df.columns)-1)

    target_names = tuple(data_df[target_col].unique())
    if target_names[0]!=pos_target:
        target_names = target_names[1], target_names[0]
    bias_names = tuple(data_df[bias_col].unique())

    n_target = [0]*len(target_names)
    for i, t in enumerate(target_names):
        n_target[i] = data_df[data_df[target_col] == t].shape[0]
        print('Number of', target_col, t, '= ', n_target[i])
    n_bias = [0]*len(bias_names)
    for i, b in enumerate(bias_names):
        n_bias[i] = data_df[data_df[bias_col] == b].shape[0]
        print('Number of {} {} = {}'.format(bias_col, b, n_bias[i]))

    columns = list(data_df)
    target_by_bias = data_df.groupby([bias_col, target_col]).count()[columns[0]]
    display(target_by_bias)

    pos_bias_frac = [0.]*len(bias_names)
    for i, b in enumerate(bias_names):
        pos_bias_frac[i] = target_by_bias[b, pos_target]
        print('Proportion {} with {} {} = {:2.2%}'.format(b, target_col, pos_target, pos_bias_frac[i]/n_bias[i]))
    return bias_names


def top_n_correlated_features(data_df, sensitive_feature, n):
    corr_df = data_df.corr()
    sex_corrs = corr_df.reindex(corr_df[sensitive_feature].abs().sort_values(ascending=False).index)[sensitive_feature]
    return sex_corrs.iloc[:n]
