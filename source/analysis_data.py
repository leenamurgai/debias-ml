""" This is the 'main' python file which calls all others to produce the
Streamlit data analysis report.
"""

################################################################################
################################################################################
################################################################################

import pandas as pd
import numpy as np
import streamlit as st

from utilities import DataParams
from utilities import write_params_to_file
from data import Data
from explore import top_n_correlated_features
from plot_results import heatmap
from oversample import Oversampler

################################################################################
################################################################################
################################################################################

st.title('debias-ml: data analysis')
st.header('1 Data Ingestion''')
st.write('')

params = DataParams()
filename = params.filename
sensitive_features  = params.sensitive_features
target_feature = params.target_feature
pos_target  = params.pos_target

data_df = pd.read_csv('../data/preprocessed/'+filename, na_values='?')
st.write('Data read successfully!')

################################################################################
################################################################################
################################################################################

st.header('2 Data Exploration & Preparation')
st.write('')
st.subheader('2.1 Peek at the raw data')
st.write('')
st.write('**The first 5 rows of the raw data:**')
st.write(data_df.head())
st.write('')
st.write('**Some basic statistics:**')
st.write('Number of data points =', len(data_df.index))
st.write('Number of features =', len(data_df.columns)-1)

################################################################################

st.write('')
st.subheader('2.2 Exploration & Processing')
st.write('')

data = Data(data_df, sensitive_features, target_feature, pos_target)
data_df = data.data_df
bias_cols = data.bias_cols
target_col = data.target_col
feature_cols = data.feature_cols
bias_col_types = data.bias_col_types

# Save our processed data
data_df.to_csv('../data/processed/'+filename, index=False)
write_params_to_file(bias_cols, target_col, bias_col_types, data.categories)

################################################################################

st.write('')
st.subheader('2.3 Post-processing exploration')
st.write('')

st.write("**Number features: **", len(feature_cols))
st.write("**Target column: **", target_col)
st.write("**Bias columns: **",bias_cols)
st.write('')
st.write("""We have reduced each sensitive feature to a single binary class.
            In each case we keep the feature that the bias appears to be in
            favour of.""")
st.write(bias_cols)
st.write('')
st.write('**The first 5 rows of the data after processing:**')
st.write(data_df.head())
st.write('')
st.write('**Top 10 most correlated features to the target feature**')
st.write('')
st.write(top_n_correlated_features(data_df, target_col, 10))
st.write('')
st.write('**Top 10 most correlated features to the bias feature**')
for b in bias_cols:
    st.write('')
    st.write(top_n_correlated_features(data_df, b, 10))
st.write('')
st.write('**Correlation Heatmap**')
corr_df = data_df.corr()
heatmap(corr_df, 'correlation-heat-map')

################################################################################

st.write('')
st.subheader('2.4 Oversample')
st.write('')

# Extract feature (X) and target (y) columns
X_all = data_df[feature_cols]
y_all = data_df[target_col]
Z_all = data_df[bias_cols]

# Oversample data
oversampler = Oversampler(X_all, y_all, Z_all, target_col, bias_cols, bias_col_types)
oversampler.original_data_stats()
X_new, y_new, Z_new = oversampler.get_oversampled_data()

st.write('Augmented data set: {} samples'.format(X_new.shape[0]))

################################################################################

st.subheader('2.5 Post-augmentation exploration')
st.write('')

new_data_df = pd.DataFrame(columns=list(data_df))
new_data_df[list(X_all)] = X_new
new_data_df[target_col] = y_new

st.write('**Top 10 most correlated features to the target feature**')
st.write('')
st.write(top_n_correlated_features(new_data_df, target_col, 10))
st.write('')
st.write('**Top 10 most correlated features to the sensitive features**')
for b in bias_cols:
    st.write('')
    st.write(top_n_correlated_features(new_data_df, b, 10))
st.write('')
st.write("""**Heatmap showing change in correlations after augmenting data by
              oversampling once**""")
heatmap(new_data_df.corr() - corr_df, 'correlation-change')

################################################################################
################################################################################
################################################################################
