""" This is the 'main' python file which calls all others to produce the
Streamlit model analysis report.
"""

################################################################################
################################################################################
################################################################################

import pandas as pd
import numpy as np
import streamlit as st

from utilities import DataParams
from utilities import ProcessedDataParams
from oversample import Oversampler
from model import nn_classifier
from train_test import make_training_and_test_sets
from train_test import normalise
from train_test import make_results_df
from train_test import train_predict
from plot_results import probability_density_functions

################################################################################
################################################################################
################################################################################

st.title('debias-ml: model testing')
st.header('1 Processed Data Ingestion''')
st.write('')

params = DataParams()
filename = params.filename
sensitive_features  = params.sensitive_features
target_feature = params.target_feature
pos_target  = params.pos_target
n_train = params.n_train

data_df = pd.read_csv('../data/processed/'+filename)
st.write('Data read successfully!')

params = ProcessedDataParams()
bias_cols = params.bias_cols
target_col = params.target_col
feature_cols = list(data_df.columns)
feature_cols.remove(target_col)
bias_col_types = params.bias_col_types
categories = params.categories

################################################################################
################################################################################
################################################################################

st.header('2 Process Data for Training''')
st.write('')
st.subheader('2.1 Data Exploration')
st.write('')

# Extract feature (X) and target (y) columns
st.write("Number features: ", len(feature_cols))
st.write("Target column: ", target_col)
st.write("Bias columns: ",bias_cols)

X_all = data_df[feature_cols]
y_all = data_df[target_col]
Z_all = data_df[bias_cols]

################################################################################

st.write('')
st.subheader('2.2 Splitting data into training and test sets')
st.write('')

# Splitting the original dataset into training and testing parts
(X_train, X_train2, X_train1, X_test,
y_train, y_train2, y_train1, y_test,
Z_train, Z_test) = make_training_and_test_sets(X_all, y_all, Z_all, n_train)
X_train, X_train2, X_train1, X_test = normalise(X_train,  X_train2,  X_train1, X_test)

st.write('Training set: {} samples'.format(X_train.shape[0]))
st.write('Test set: {} samples'.format(X_test.shape[0]))

################################################################################

st.write('')
st.subheader('2.3 Setup the Oversampler')
st.write('')

# Set up the Oversampler
oversampler = Oversampler(X_train, y_train, Z_train,
                          target_col, bias_cols, bias_col_types)
oversampler.original_data_stats()
X_new, y_new, Z_new = oversampler.get_oversampled_data()

st.write('Augmented data set: {} samples'.format(X_new.shape[0]))

# Work out how many data points we need to train from our augmented dataset ()
new_n_train = X_new.shape[0] * n_train / X_all.shape[0]
new_n_train = int(new_n_train - new_n_train % 3)

st.write('')
st.write('**We split our augmented data set into training and test sets:**')

(X_train_new, X_train2_new, X_train1_new, X_test_new,
y_train_new, y_train2_new, y_train1_new, y_test_new,
Z_train_new, Z_test_new) = make_training_and_test_sets(X_new, y_new, Z_new, new_n_train)

st.write('Augmented training set: {} samples'.format(X_train_new.shape[0]))
st.write('Augmented test set: {} samples'.format(X_test_new.shape[0]))

################################################################################
################################################################################
################################################################################

st.header('3 Training a 3 layer neural network...')
st.write('')
st.subheader('3.1 ...on all the data')
st.write('')

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train.shape[1])
results_df = make_results_df(n_train)
# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1, y_train1, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train2, y_train2, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train, y_train, X_test, y_test, results_df)
st.write(results_df)
probability_density_functions(y_pred, Z_test,
                              target_feature, sensitive_features, bias_cols,
                              categories, 'all-data')

################################################################################

st.write('')
st.subheader('3.2 ...with bias columns removed')
st.write('')

clf_nn = nn_classifier(n_features=X_train[X_train.columns.difference(bias_cols)].shape[1])
# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn,
                       X_train1[X_train1.columns.difference(bias_cols)],
                       y_train1, X_test[X_test.columns.difference(bias_cols)],
                       y_test,
                       results_df)
y_pred = train_predict(clf_nn,
                       X_train2[X_train2.columns.difference(bias_cols)],
                       y_train2, X_test[X_test.columns.difference(bias_cols)],
                       y_test,
                       results_df)
y_pred = train_predict(clf_nn,
                       X_train[X_train.columns.difference(bias_cols)],
                       y_train,
                       X_test[X_test.columns.difference(bias_cols)],
                       y_test,
                       results_df)
st.write(results_df)
probability_density_functions(y_pred, Z_test,
                              target_feature, sensitive_features, bias_cols,
                              categories, 'no-bias-data')

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
results_df = make_results_df(new_n_train)
# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1_new, y_train1_new,
                               X_test_new, y_test_new, results_df)
y_pred = train_predict(clf_nn, X_train2_new, y_train2_new,
                               X_test_new, y_test_new, results_df)
y_pred = train_predict(clf_nn, X_train_new, y_train_new,
                               X_test_new, y_test_new, results_df)
st.write(results_df)
probability_density_functions(y_pred, Z_test_new,
                              target_feature, sensitive_features, bias_cols,
                              categories,'fair-data')

################################################################################

st.write('')
st.subheader('3.4 ...after oversampling testing on original test data')
st.write('')

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train_new.shape[1])

# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1_new, y_train1_new,
                               X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train2_new, y_train2_new,
                               X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train_new, y_train_new,
                               X_test, y_test, results_df)
st.write(results_df)
probability_density_functions(y_pred, Z_test,
                              target_feature, sensitive_features, bias_cols,
                              categories, 'fair-algo')

################################################################################
################################################################################
################################################################################
