import pandas as pd

def preprocess_data(data_df):
    out = pd.DataFrame(index=data_df.index)  # output dataframe, initially empty
    categories = {}
    col_df = pd.DataFrame()
    # Check each column
    for col, col_data in data_df.iteritems():
        # If non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col, prefix_sep=' is ')
            categories[col] = list(col_data)
        out = out.join(col_data)  # collect column(s) in output dataframe
    return out.fillna('Unknown'), categories


def remove_redundant_cols(data_df, categories, target_col, pos_target):
    for k, v in categories.items():
        col = data_df[v].sum().idxmin()
        if col == (target_col+' is '+pos_target):
            col = data_df[v].sum().idxmax()
        v.remove(col)
        del data_df[col]
