################################################################################
################################################################################
################################################################################
import sys
import pandas as pd
import streamlit as st
################################################################################
################################################################################
################################################################################
st.write('# Eliminating Bias in Machine Learning')
st.write('## 0. Ingest data''')
data_df = pd.read_csv('../data/preprocessed/adult-data.csv', na_values='?')
st.write('')
st.write('### Data read successfully!')
################################################################################
################################################################################
################################################################################
from explore import basic_stats

st.write('## 1. Exploration')
st.write('')
st.write('**The first 5 rows of the data:**')
st.write(data_df.head())

bias_name = 'sex'
target_name = 'ann_salary'
pos_target = '>50K'
basic_stats(data_df, bias_name, target_name, pos_target)
################################################################################
################################################################################
################################################################################
from data import preprocess_data
from data import remove_redundant_cols
from explore import top_n_correlated_features
from plots import heatmap
from train_test import make_training_and_test_sets
from train_test import normalise
from oversample import oversample

st.write('## 2. Preparing the Data')
st.write('')
st.write('### 2.1 Converting categorical columns to binary')
st.write('')

prefix_sep = ' is '
data_df, categories = preprocess_data(data_df, prefix_sep)
remove_redundant_cols(data_df, categories, target_name, pos_target)
bias_col = bias_name + prefix_sep + categories[bias_name][1]
target_col = target_name + prefix_sep + categories[target_name][1]

st.write('**The first 5 rows of the data:**')
st.write(data_df.head())
st.write('')

st.write('### 2.2 Post-processing exploration')
st.write('')
st.write('**Top 10 most correlated feature to the target feature**')
st.write('')
st.write(top_n_correlated_features(data_df, target_col, 10))
st.write('')
st.write('**Top 10 most correlated feature to the bias feature**')
st.write('')
st.write(top_n_correlated_features(data_df, bias_col, 10))
st.write('')
st.write('**Correlation Heatmap**')
corr_df = data_df.corr()
heatmap(corr_df, 'correlation-heat-map')

st.write('')
st.write('### 2.3 Separate features and labels')
st.write('')

# Extract feature (X) and target (y) columns
feature_cols = list(data_df.columns)
feature_cols.remove(target_col) # leave bias_col in features

st.write("Number feature columns: ", len(feature_cols))
st.write("Target column: ",target_col)
st.write("Bias column: ",bias_col)

X_all = data_df[feature_cols]
y_all = data_df[target_col]
z_all = data_df[bias_col]

st.write('')
st.write('''### 2.4 Splitting data into training and test sets''')
st.write('')

# Splitting the original dataset into training and testing parts
n_train = 30000
X_train, X_train2, X_train1, X_test, y_train, y_train2, y_train1, y_test, z_train, z_test = make_training_and_test_sets(X_all, y_all, z_all, n_train)
X_train, X_train2, X_train1, X_test = normalise(X_train,  X_train2,  X_train1,  X_test)

st.write('Training set: {} samples'.format(X_train.shape[0]))
st.write('Test set: {} samples'.format(X_test.shape[0]))

st.write('')
st.write('''### 2.5 Augmenting the training data by oversampling''')
st.write('')

# Oversampling to address bias in the training dataset
X_new, y_new, z_new = oversample(X_train, y_train, z_train, target_col, categories[bias_name])

# Work out how many data point we need to train from our augmented dataset ()
new_n_train = X_new.shape[0]*n_train/X_all.shape[0]
new_n_train = int(new_n_train - new_n_train%3)

X_train_new, X_train2_new, X_train1_new, X_test_new, y_train_new, y_train2_new, y_train1_new, y_test_new, z_train_new, z_test_new = make_training_and_test_sets(X_new, y_new, z_new, new_n_train)

st.write('')
st.write('Augmented training set: {} samples'.format(X_train_new.shape[0]))
st.write('Augmented test set: {} samples'.format(X_test_new.shape[0]))
#st.write(bias_name, categories[bias_name])
#st.write(target_name, categories[target_name])

new_data_df = pd.DataFrame(columns=list(data_df))
new_data_df[list(X_train)] = X_train_new
new_data_df[target_col] = y_train_new
st.write('')
st.write('**Heatmap showing change in correlations after augmenting data by oversampling**')
heatmap(new_data_df.corr()-corr_df, 'correlation-change')
################################################################################
################################################################################
################################################################################
from train_test import make_results_df
from model import nn_classifier
from train_test import train_predict
from plots import probability_density_function
from plots import get_bias_factor

st.write('## 3 Training a 3 layer neural network...')
st.write('')
st.write('### 3.1 ...on all the data')
st.write('')

results_df = make_results_df(n_train)

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train.shape[1])
#st.write(clf_nn)

# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1, y_train1, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train2, y_train2, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train, y_train, X_test, y_test, results_df)

st.write(results_df)
probability_density_function(y_pred, z_test, target_col, bias_name, categories[bias_name], 'all-data-dist')
get_bias_factor(y_pred, z_test, target_name, bias_name, categories, 'all-data-hist')

st.write('')
st.write('### 3.2 ...with gender information removed')
st.write('')

clf_nn = nn_classifier(n_features=X_train[X_train.columns.difference([bias_col])].shape[1])
#st.write(clf_nn)

# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1[X_train1.columns.difference([bias_col])], y_train1, X_test[X_test.columns.difference([bias_col])], y_test, results_df)
y_pred = train_predict(clf_nn, X_train2[X_train2.columns.difference([bias_col])], y_train2, X_test[X_test.columns.difference([bias_col])], y_test, results_df)
y_pred = train_predict(clf_nn, X_train[X_train.columns.difference([bias_col])], y_train, X_test[X_test.columns.difference([bias_col])], y_test, results_df)

st.write(results_df)
probability_density_function(y_pred, z_test, target_col, bias_name, categories[bias_name], 'no-sex-data-dist')
get_bias_factor(y_pred, z_test, target_name, bias_name, categories, 'no-sex-data-hist')

st.write('')
st.write('### 3.3 ...after oversampling well paid women in the training data and testing on similarly oversampled data')
st.write('')
st.write('Here we want to validate our oversampling - if we have done it correctly, when we test on data from the same distribution we should find that the bias reduction is significant with a bias factor close to 1.')
st.write('')

results_df = make_results_df(new_n_train)

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train_new.shape[1])
#st.write(clf_nn)

# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1_new, y_train1_new, X_test_new, y_test_new, results_df)
y_pred = train_predict(clf_nn, X_train2_new, y_train2_new, X_test_new, y_test_new, results_df)
y_pred = train_predict(clf_nn, X_train_new, y_train_new, X_test_new, y_test_new, results_df)

st.write(results_df)
probability_density_function(y_pred, z_test_new, target_col, bias_name, categories[bias_name], 'fair-data-dist')
get_bias_factor(y_pred, z_test_new, target_name, bias_name, categories, 'fair-data-hist')

st.write('')
st.write('### 3.4 ...after oversampling well paid women in the training data and testing on original test data')
st.write('')

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train_new.shape[1])
#st.write(clf_nn)

# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1_new, y_train1_new, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train2_new, y_train2_new, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train_new, y_train_new, X_test, y_test, results_df)

st.write(results_df)
probability_density_function(y_pred, z_test, target_col, bias_name, categories[bias_name], 'fair-algo-dist')
get_bias_factor(y_pred, z_test, target_name, bias_name, categories, 'fair-algo-hist')
################################################################################
################################################################################
################################################################################
