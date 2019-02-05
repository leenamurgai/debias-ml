################################################################################
################################################################################
################################################################################
import sys
import pandas as pd
import streamlit as st
################################################################################
################################################################################
################################################################################
st.title('Eliminating Bias in Machine Learning')
st.header('0 Ingest data''')
data_df = pd.read_csv('../data/preprocessed/adult-data.csv', na_values='?')
st.write('')
st.write('Data read successfully!')
################################################################################
################################################################################
################################################################################
from explore import basic_stats

st.write('1 Exploration')
st.write('')
st.write('**The first 5 rows of the data:**')
st.write(data_df.head())

bias_names  = ['sex', 'race']
target_name = 'ann_salary'
pos_target  = '>50K'
# We transform each bias_name to a bianry data type
# pos_bias_labels is a dict mapping each bias_name to a list of the types
# we associate to the postive label
pos_bias_labels = basic_stats(data_df, bias_names, target_name, pos_target)
#st.write(pos_bias_labels)
################################################################################
################################################################################
################################################################################
from data import preprocess_data
from data import binarise_bias_cols
from data import remove_redundant_cols
from data import move_target_col_to_end
from data import categories_to_columns
from explore import top_n_correlated_features
from plots import heatmap
from train_test import make_training_and_test_sets
from train_test import normalise
from oversample import oversample

st.header('2 Preparing the Data')
st.write('')
st.subheader('2.1 Converting categorical columns to binary')

data_df, categories = preprocess_data(data_df)
binarise_bias_cols(data_df, categories, pos_bias_labels)
remove_redundant_cols(data_df, categories, target_name, pos_target)
categories_col   = categories_to_columns(categories)
bias_col_types   = [categories[b] for b in bias_names]
bias_cols        = [categories_col[b][1] for b in bias_names]
target_col_types = categories[target_name]
target_col       = categories_col[target_name][1]
move_target_col_to_end(data_df, target_col)

st.write('')
st.write('We also reduce our bias features to 2 possible classes so our binary bias features are')
st.write(bias_cols)

st.write('')
st.write('**The first 5 rows of the data:**')
st.write(data_df.head())
st.write('')

st.subheader('2.2 Post-processing exploration')
st.write('')
st.write('**Top 10 most correlated feature to the target feature**')
st.write('')
st.write(top_n_correlated_features(data_df, target_col, 10))
st.write('')
st.write('**Top 10 most correlated feature to the bias feature**')
for b in bias_cols:
    st.write('')
    st.write(top_n_correlated_features(data_df, b, 10))
st.write('')
st.write('**Correlation Heatmap**')
corr_df = data_df.corr()
heatmap(corr_df, 'correlation-heat-map')

st.write('')
st.subheader('2.3 Separate features and labels')
st.write('')

# Extract feature (X) and target (y) columns
feature_cols = list(data_df.columns)
feature_cols.remove(target_col) # leave bias_col in features

st.write("Number feature columns: ", len(feature_cols))
st.write("Target column: ",target_col)
st.write("Bias columns: ",bias_cols)

X_all = data_df[feature_cols]
y_all = data_df[target_col]
Z_all = data_df[bias_cols]

st.write('')
st.subheader('2.4 Splitting data into training and test sets')
st.write('')

# Splitting the original dataset into training and testing parts
n_train = 30000
X_train, X_train2, X_train1, X_test, y_train, y_train2, y_train1, y_test, Z_train, Z_test = make_training_and_test_sets(X_all, y_all, Z_all, n_train)
X_train, X_train2, X_train1, X_test = normalise(X_train,  X_train2,  X_train1,  X_test)

st.write('Training set: {} samples'.format(X_train.shape[0]))
st.write('Test set: {} samples'.format(X_test.shape[0]))

st.write('')
st.subheader('2.5 Augmenting the training data by oversampling')
st.write('')

# Oversampling to address bias in the training dataset
X_new, y_new, Z_new = oversample(X_train, y_train, Z_train, target_col, bias_cols, bias_col_types)
st.write('')
st.write('Augmented data set: {} samples'.format(X_new.shape[0]))

# Work out how many data point we need to train from our augmented dataset ()
new_n_train = X_new.shape[0]*n_train/X_all.shape[0]
new_n_train = int(new_n_train - new_n_train%3)

st.write('')
st.write('**We split our augmented data set into training and test sets:**')
X_train_new, X_train2_new, X_train1_new, X_test_new, y_train_new, y_train2_new, y_train1_new, y_test_new, Z_train_new, Z_test_new = make_training_and_test_sets(X_new, y_new, Z_new, new_n_train)

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
from plots import probability_density_functions
from plots import get_bias_factor

st.header('3 Training a 3 layer neural network...')
st.write('')
st.subheader('3.1 ...on all the data')
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
probability_density_functions(y_pred, Z_test, target_name, bias_names, categories, 'all-data')


st.write('')
st.subheader('3.2 ...with gender and race information removed')
st.write('')

clf_nn = nn_classifier(n_features=X_train[X_train.columns.difference(bias_cols)].shape[1])
#st.write(clf_nn)

# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1[X_train1.columns.difference(bias_cols)], y_train1, X_test[X_test.columns.difference(bias_cols)], y_test, results_df)
y_pred = train_predict(clf_nn, X_train2[X_train2.columns.difference(bias_cols)], y_train2, X_test[X_test.columns.difference(bias_cols)], y_test, results_df)
y_pred = train_predict(clf_nn, X_train[X_train.columns.difference(bias_cols)], y_train, X_test[X_test.columns.difference(bias_cols)], y_test, results_df)

st.write(results_df)
probability_density_functions(y_pred, Z_test, target_name, bias_names, categories, 'no-bias-data')

st.write('')
st.subheader('3.3 ...after oversampling well paid women in the training data and testing on similarly oversampled data')
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
probability_density_functions(y_pred, Z_test_new, target_name, bias_names, categories, 'fair-data')

st.write('')
st.subheader('3.4 ...after oversampling well paid women in the training data and testing on original test data')
st.write('')

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train_new.shape[1])
#st.write(clf_nn)

# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1_new, y_train1_new, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train2_new, y_train2_new, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train_new, y_train_new, X_test, y_test, results_df)

st.write(results_df)
probability_density_functions(y_pred, Z_test, target_name, bias_names, categories, 'fair-algo')

################################################################################
################################################################################
################################################################################
