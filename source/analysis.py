""" This is the 'main' python file which calls all others to produce the
Streamlit report.
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

from model import nn_classifier
from train_test import make_training_and_test_sets
from train_test import normalise
from train_test import make_results_df
from train_test import make_train_test_sets
from train_test import train_predict
from train_test import train_predict_new
from plot_results import probability_density_functions
from plot_results import plot_distributions

from sklearn.utils import shuffle

################################################################################
################################################################################
################################################################################

st.title('debias-ml')
st.header('1 Data Ingestion''')
st.write('')

params = DataParams()
filename = params.filename
sensitive_features  = params.sensitive_features
target_feature = params.target_feature
pos_target  = params.pos_target
n_train = params.n_train

data_df = pd.read_csv('../data/preprocessed/'+filename, na_values='?')
st.write('Data read successfully!')

################################################################################
################################################################################
################################################################################

st.header('2 Data Exploration & Preparation')
st.write('')
st.subheader('2.1 Peek at the raw data')

st.write('**The first 5 rows of the raw data:**')
st.write(data_df.head())
st.write('')
st.write('**Some basic statistics:**')
st.write('Number of data points =', len(data_df.index))
st.write('Number of features =', len(data_df.columns)-1)

################################################################################

st.write('')
st.subheader('2.2 Exploration & Processing')

data = Data(data_df, sensitive_features, target_feature, pos_target)
data_df = data.data_df
bias_cols = data.bias_cols
target_col = data.target_col
feature_cols = data.feature_cols
bias_col_types = data.bias_col_types
categories = data.categories

# Save our processed data
data_df.to_csv('../data/processed/'+filename, index=False)
write_params_to_file(bias_cols, target_col, bias_col_types, categories)

################################################################################

st.write('')
st.subheader('2.3 Post-processing exploration')
st.write('')
# Extract feature (X) and target (y) columns
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
st.subheader('2.4 Splitting data into training and test sets')
st.write('')

X_all = data_df[feature_cols]
y_all = data_df[target_col]
Z_all = data_df[bias_cols]

# Splitting the original dataset into training and testing parts
(X_train, X_train2, X_train1, X_test,
y_train, y_train2, y_train1, y_test,
Z_train, Z_test) = make_training_and_test_sets(X_all, y_all, Z_all, n_train)
X_train, X_train2, X_train1, X_test = normalise(X_train,  X_train2,  X_train1, X_test)

st.write('Training set: {} samples'.format(X_train.shape[0]))
st.write('Test set: {} samples'.format(X_test.shape[0]))

################################################################################

st.write('')
st.subheader('2.5 Setup the Oversampler')
st.write('')

# Set up the Oversampler
oversampler = Oversampler(X_train, y_train, Z_train,
                          target_col, bias_cols, bias_col_types)
oversampler.original_data_stats()

X_new, y_new, Z_new = oversampler.get_oversampled_data()
st.write('')
st.write('Augmented data set: {} samples'.format(X_new.shape[0]))

# Work out how many data point we need to train from our augmented dataset ()
new_n_train = X_new.shape[0]*n_train/X_all.shape[0]
new_n_train = int(new_n_train - new_n_train%3)

st.write('')
st.write('**We split our augmented data set into training and test sets:**')
(X_train_new, X_train2_new, X_train1_new, X_test_new,
y_train_new, y_train2_new, y_train1_new, y_test_new,
Z_train_new, Z_test_new) = make_training_and_test_sets(X_new, y_new, Z_new, new_n_train)

st.write('Augmented training set: {} samples'.format(X_train_new.shape[0]))
st.write('Augmented test set: {} samples'.format(X_test_new.shape[0]))

################################################################################

st.subheader('2.7 Post-aumentation exploration')
st.write('')

new_data_df = pd.DataFrame(columns=list(data_df))
new_data_df[list(X_train)] = X_train_new
new_data_df[target_col] = y_train_new

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
              oversampling**""")
heatmap(new_data_df.corr() - corr_df, 'correlation-change')

################################################################################
################################################################################
################################################################################

st.header('3 Training a 3 layer neural network...')
st.write('')
st.subheader('3.1 ...on all the data')
st.write('')

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train.shape[1])
# Get distributions for slides
results_df = pd.DataFrame()
y_pred = train_predict_new(clf_nn, X_train, y_train, X_test, y_test,
                                   results_df, 0)
plot_distributions(y_pred, Z_test, target_feature, sensitive_features,
                   bias_cols, categories, 0, results_df, 'all-data')

################################################################################

st.write('')
st.subheader('3.2 ...with bias columns removed')
st.write('')

clf_nn = nn_classifier(n_features=X_train[X_train.columns.difference(bias_cols)].shape[1])
# Get distributions for slides
results_df = pd.DataFrame()
y_pred = train_predict_new(clf_nn,
                           X_train[X_train.columns.difference(bias_cols)],
                           y_train,
                           X_test[X_test.columns.difference(bias_cols)],
                           y_test,
                           results_df, 0)
plot_distributions(y_pred, Z_test, target_feature, sensitive_features,
                   bias_cols, categories, 0, results_df, 'no-bias-data')

################################################################################

st.write('')
st.subheader("""3.3 ...after oversampling and testing on the oversampled data""")
st.write('')
st.write("""These results do not reflect how our model would work on real data
            since the test set is from the oversampled data. The purpose of
            these tests is to validate our oversampling - if we have done it
            correctly, when we test on data from the same (oversampled)
            distribution we should find that the bias reduction is significant
            with a bias factor close to 1.""")
st.write('')

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train_new.shape[1])
# Get distributions for slides
results_df = pd.DataFrame()
y_pred = train_predict_new(clf_nn, X_train_new, y_train_new,
                                   X_test_new, y_test_new, results_df, 0)
plot_distributions(y_pred, Z_test_new,  target_feature, sensitive_features,
                   bias_cols, categories, 0, results_df, 'fair-data')

################################################################################

st.write('')
st.subheader("""3.4 ...after oversampling by different amounts""")
st.write('')

results_df = pd.DataFrame()

#for factor in np.linspace(0.0, 5.0, num=11):
for factor in range(1, 11):
    st.write('**Oversample factor:**', factor)
    # Oversampling to address bias in the training dataset
    X_new, y_new, Z_new = oversampler.get_oversampled_data(factor)
    # Shuffle the data after oversampling
    X_train_new, y_train_new, Z_train_new = shuffle(X_new, y_new, Z_new,
                                                    random_state=0)
    # initialise NeuralNet Classifier
    clf_nn = nn_classifier(n_features=X_train_new.shape[1])
    # make predictions on the test set
    y_pred = train_predict_new(clf_nn, X_train_new, y_train_new,
                                       X_test, y_test, results_df, factor)
    plot_distributions(y_pred, Z_test, target_feature, sensitive_features,
                       bias_cols, categories, factor, results_df,
                       'fair-algo-'+str(factor))

st.table(results_df)

################################################################################
################################################################################
################################################################################
