import pandas as pd
import streamlit as st

"""
class Data(object):
    def __init__(self, data_df, bias_names=['sex', 'race'], target_name='ann_salary', pos_target='>50K'):
        self.data_df, self.categories = preprocess_data(data_df)
        self.binarise_bias_cols(self.data_df, self.categories, pos_bias_labels)
        self.remove_redundant_cols(data_df, self.categories, target_name, pos_target)
        categories_col   = categories_to_columns(categories)
        bias_col_types   = [categories[b] for b in bias_names]
        bias_cols        = [categories_col[b][1] for b in bias_names]
        target_col_types = categories[target_name]
        target_col       = categories_col[target_name][1]
        move_target_col_to_end(data_df, target_col)
"""

def preprocess_data(data_df, prefix_sep=' is '):
    out = pd.DataFrame(index=data_df.index)  # output dataframe, initially empty
    categories = {}
    for col, col_data in data_df.iteritems():
        # If non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            categories[col] = list(data_df[col].fillna('Unknown').unique())
            col_data = pd.get_dummies(col_data, prefix=col, prefix_sep=prefix_sep)
        out = out.join(col_data)
    [v.remove('Unknown') for v in categories.values() if 'Unknown' in v]
    return out.fillna('Unknown'), categories


def categories_to_columns(categories, prefix_sep=' is '):
    categories_col = {}
    for k, v in categories.items():
        val = [k + prefix_sep + vi for vi in v]
        categories_col[k] = val
    return categories_col


def binarise_bias_cols(data_df, categories, pos_bias_labels, prefix_sep=' is '):
    categories_col = categories_to_columns(categories, prefix_sep)
    pos_bias_cols = {}
    for key in pos_bias_labels.keys():
        pos_bias_cols[key] = [key + prefix_sep + name for name in pos_bias_labels[key]]
        if len(pos_bias_labels[key])>1:
            new_col1 = key + prefix_sep
            new_col0 = key + prefix_sep + 'not '
            for i in pos_bias_labels[key]:
                new_col1 = new_col1 + i + '-'
                new_col0 = new_col0 + i + '-'
            new_col1 = new_col1[:-1]
            new_col0 = new_col0[:-1]
            for i in data_df.index:
                data_df.at[i, new_col1] = 0
                data_df.at[i, new_col0] = 0
                if data_df[pos_bias_cols[key]].loc[i].sum()==1:
                    data_df.at[i, new_col1] = 1
                else:
                    data_df.at[i, new_col0] = 1
            for col in categories_col[key]:
                del data_df[col]
            categories[key] = [new_col0[len(key)+4:], new_col1[len(key)+4:]]


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


def move_target_col_to_end(data_df1, target_col):
    target_df = data_df1[target_col]
    del data_df1[target_col]
    data_df1[target_col] = target_df
