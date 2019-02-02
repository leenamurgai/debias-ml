import pandas as pd

def preprocess_data(data_df):
    out = pd.DataFrame(index=data_df.index)  # output dataframe, initially empty
    categories = {}
    col_df = pd.DataFrame()
    # Check each column
    for col, col_data in data_df.iteritems():
        # If non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            categories[col] = list(data_df[col].fillna('Unknown').unique())
            col_data = pd.get_dummies(col_data, prefix=col, prefix_sep=' is ')
        out = out.join(col_data)  # collect column(s) in output dataframe
    [v.remove('Unknown') for v in categories.values() if 'Unknown' in v]
    return out.fillna('Unknown'), categories

def remove_redundant_cols(data_df, categories, target_col, pos_target):
    for k, v in categories.items():
        val = [k+' is '+vi for vi in v]
        col = data_df[val].sum().idxmin()
        if col == (target_col+' is '+pos_target):
            col = data_df[val].sum().idxmax()
        the_val = col[(len(k)+4):]
        if v[0]!=the_val:
            v.remove(the_val)
            v.insert(0,the_val)
        del data_df[col]
