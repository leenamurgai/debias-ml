""" This is the 'main' python file which calls all others to produce the
Streamlit report.
"""

################################################################################
################################################################################
################################################################################

import pandas as pd
import numpy as np
import streamlit as st

################################################################################
################################################################################
################################################################################

st.title('Eliminating Bias in Machine Learning')
st.header('0 Ingest data''')

filename = 'adult-data.csv'

data_df = pd.read_csv('../data/preprocessed/'+filename, na_values='?')
st.write('')
st.write('Data read successfully!')

################################################################################
################################################################################
################################################################################

sensitive_features  = ['sex', 'race']
target_feature = 'ann_salary'
pos_target  = '>50K'

st.header('1 Exploration')
st.write('')

st.write('**The first 5 rows of the raw data:**')
st.write(data_df.head())

st.write('')
st.write('**Some basic statistics:**')
st.write('Number of data points =', len(data_df.index))
st.write('Number of features =', len(data_df.columns)-1)

################################################################################
################################################################################
################################################################################

from data import Data
from explore import top_n_correlated_features
from plots import heatmap
from train_test import make_training_and_test_sets
from train_test import normalise
from oversample import Oversampler

st.header('2 Preparing the Data')
st.write('')
st.subheader('2.1 Converting categorical columns to binary')

data = Data(data_df, sensitive_features, target_feature, pos_target)

st.write('')
st.write("""We reduce our bias features to 2 possible classes so our bias
features each correspond to a single column""")
st.write(data.bias_cols)

st.write('')
st.write('**The first 5 rows of the data:**')
st.write(data.data_df.head())

# Save our processed data
data.data_df.to_csv('../data/processed/'+filename, index=False)

################################################################################

st.write('')
st.subheader('2.2 Post-processing exploration')
st.write('')
st.write('**Top 10 most correlated features to the target feature**')
st.write('')
st.write(top_n_correlated_features(data.data_df, data.target_col, 10))
st.write('')
st.write('**Top 10 most correlated features to the bias feature**')
for b in data.bias_cols:
    st.write('')
    st.write(top_n_correlated_features(data.data_df, b, 10))
st.write('')
st.write('**Correlation Heatmap**')
corr_df = data.data_df.corr()
heatmap(corr_df, 'correlation-heat-map')

################################################################################

st.write('')
st.subheader('2.3 Separate features and labels')
st.write('')

# Extract feature (X) and target (y) columns
st.write("Number features: ", len(data.feature_cols))
st.write("Target column: ", data.target_col)
st.write("Bias columns: ",data.bias_cols)

X_all = data.data_df[data.feature_cols]
y_all = data.data_df[data.target_col]
Z_all = data.data_df[data.bias_cols]

################################################################################

st.write('')
st.subheader('2.4 Splitting data into training and test sets')
st.write('')

# Splitting the original dataset into training and testing parts
n_train = 30000
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
                          data.target_col, data.bias_cols, data.bias_col_types)
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

st.subheader('2.6 Post-aumentation exploration')
st.write('')

new_data_df = pd.DataFrame(columns=list(data.data_df))
new_data_df[list(X_train)] = X_train_new
new_data_df[data.target_col] = y_train_new

st.write('**Top 10 most correlated features to the target feature**')
st.write('')
st.write(top_n_correlated_features(new_data_df, data.target_col, 10))
st.write('')
st.write('**Top 10 most correlated features to the bias feature**')
for b in data.bias_cols:
    st.write('')
    st.write(top_n_correlated_features(new_data_df, b, 10))
st.write('')
st.write("""**Heatmap showing change in correlations after augmenting data by
              oversampling**""")
heatmap(new_data_df.corr()-corr_df, 'correlation-change')

################################################################################
################################################################################
################################################################################
from train_test import make_results_df
from model import nn_classifier
from train_test import make_train_test_sets
from train_test import train_predict
from train_test import train_predict_new
from plots import probability_density_functions
from plots import plot_distributions

st.header('3 Training a 3 layer neural network...')
st.write('')

st.subheader('3.1 ...on all the data')
st.write('')

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train.shape[1])
"""
results_df = make_results_df(n_train)
# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1, y_train1, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train2, y_train2, X_test, y_test, results_df)
y_pred = train_predict(clf_nn, X_train, y_train, X_test, y_test, results_df)
st.table(results_df)
#probability_density_functions(y_pred, Z_test, data, all-data')
"""
# Get distributions for slides
results_df = pd.DataFrame()
y_pred = train_predict_new(clf_nn, X_train, y_train, X_test, y_test,
                                   results_df, 0)
plot_distributions(y_pred, Z_test, data, 0, results_df, 'all-data')

################################################################################

st.write('')
st.subheader('3.2 ...with bias columns removed')
st.write('')

clf_nn = nn_classifier(n_features=X_train[X_train.columns.difference(data.bias_cols)].shape[1])
"""
# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn,
                       X_train1[X_train1.columns.difference(data.bias_cols)],
                       y_train1, X_test[X_test.columns.difference(data.bias_cols)],
                       y_test,
                       results_df)
y_pred = train_predict(clf_nn,
                       X_train2[X_train2.columns.difference(data.bias_cols)],
                       y_train2, X_test[X_test.columns.difference(data.bias_cols)],
                       y_test,
                       results_df)
y_pred = train_predict(clf_nn,
                       X_train[X_train.columns.difference(data.bias_cols)],
                       y_train,
                       X_test[X_test.columns.difference(data.bias_cols)],
                       y_test,
                       results_df)
st.table(results_df)
#probability_density_functions(y_pred, Z_test, data, 'no-bias-data')
"""
# Get distributions for slides
results_df = pd.DataFrame()
y_pred = train_predict_new(clf_nn,
                           X_train[X_train.columns.difference(data.bias_cols)],
                           y_train,
                           X_test[X_test.columns.difference(data.bias_cols)],
                           y_test,
                           results_df, 0)
plot_distributions(y_pred, Z_test, data, 0, results_df, 'no-bias-data')

################################################################################

st.write('')
st.subheader("""3.3 ...after oversampling and testing on the oversampled data""")
st.write('')
st.write("""These results do not reflect how our model would work on real data
    since the test set is from the oversampled data. The purpose of these tests
    is to validate our oversampling - if we have done it correctly, when we test
    on data from the same (oversampled) distribution we should find that the
    bias reduction is significant with a bias factor close to 1.""")
st.write('')

results_df = make_results_df(new_n_train)

# initialise NeuralNet Classifier
clf_nn = nn_classifier(n_features=X_train_new.shape[1])
"""
# Train on different size training sets and predict on a separate test set
y_pred = train_predict(clf_nn, X_train1_new, y_train1_new,
                               X_test_new, y_test_new, results_df)
y_pred = train_predict(clf_nn, X_train2_new, y_train2_new,
                               X_test_new, y_test_new, results_df)
y_pred = train_predict(clf_nn, X_train_new, y_train_new,
                               X_test_new, y_test_new, results_df)
st.table(results_df)
#probability_density_functions(y_pred, Z_test_new, data,'fair-data')
"""
# Get distributions for slides
results_df = pd.DataFrame()
y_pred = train_predict_new(clf_nn, X_train_new, y_train_new,
                                   X_test_new, y_test_new, results_df, 0)
plot_distributions(y_pred, Z_test_new,  data, 0, results_df, 'fair-data')

################################################################################

"""
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
st.table(results_df)
#probability_density_functions(y_pred, Z_test, data, 'fair-algo')

# Get distributions for slides
results_df = pd.DataFrame()
y_pred = train_predict_new(clf_nn, X_train_new, y_train_new,
                                   X_test, y_test, results_df, 0)
plot_distributions(y_pred, Z_test_new, data, 0, results_df, 'fair-algo')
"""
################################################################################

from sklearn.utils import shuffle

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
    plot_distributions(y_pred, Z_test, data,
                       factor, results_df, 'fair-algo-'+str(factor))

st.table(results_df)

################################################################################
################################################################################
################################################################################
