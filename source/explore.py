def top_n_correlated_features(data_df, sensitive_feature, n):
    """returns the n features most correlated to the sensitive feature"""
    corr_df = data_df.corr()
    sex_corrs = corr_df.reindex(corr_df[sensitive_feature].abs().sort_values(ascending=False).index)[sensitive_feature]
    return sex_corrs.iloc[:n]
