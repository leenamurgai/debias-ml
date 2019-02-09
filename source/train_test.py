import time

import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


metrics = ['Training time'    ,
           'Prediction time',
           'F1 score (train)' ,
           'F1 score (test)',
           'Precision (train)',
           'Precision (test)',
           'Recall (train)'   ,
           'Recall (test)',
           'Accuracy (train)' ,
           'Accuracy (test)',
           'ROC AUC (train)'  ,
           'ROC AUC (test)']

def make_results_df(n_train):
    return pd.DataFrame(
        data = [[0.] * 3] * len(metrics),
        index = metrics,
        columns=[int(n_train/3), int(2*n_train/3), int(n_train)])


def make_training_and_test_sets(X, y, Z, num_train):
    num_all = X.shape[0]
    num_test = num_all - num_train
    test_frac = float(num_test)/float(num_all)

    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, test_size=test_frac, stratify=y, random_state=0)
    X_train2, X_train1, y_train2, y_train1 = train_test_split(X_train, y_train, test_size=0.333333, stratify=y_train, random_state=0)

    X_train = X_train.reset_index(drop=True)
    X_train2 = X_train2.reset_index(drop=True)
    X_train1 = X_train1.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_train2 = y_train2.reset_index(drop=True)
    y_train1 = y_train1.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    Z_train = Z_train.reset_index(drop=True)
    Z_test = Z_test.reset_index(drop=True)

    return X_train, X_train2, X_train1, X_test, y_train, y_train2, y_train1, y_test, Z_train, Z_test


def make_train_test_sets(X, y, Z, num_train):
    num_all = X.shape[0]
    num_test = num_all - num_train
    test_frac = float(num_test)/float(num_all)

    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, test_size=test_frac, stratify=y, random_state=0)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    Z_train = Z_train.reset_index(drop=True)
    Z_test = Z_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test, Z_train, Z_test


def normalise(X_train,  X_train2,  X_train1,  X_test):
    scaler = StandardScaler().fit(X_train.astype(float)) # scale based on X_train
    scale_func = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train  = scale_func(X_train.astype(float),  scaler)
    X_test   = scale_func(X_test.astype(float),   scaler)
    X_train2 = scale_func(X_train2.astype(float), scaler)
    X_train1 = scale_func(X_train1.astype(float), scaler)
    return X_train, X_train2, X_train1, X_test


def train_classifier(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train.values, y_train.values, epochs=20, verbose=0)
    end = time.time()
    return end-start


def predict_labels(clf, X, y):
    start = time.time()
    y_pred = pd.Series(clf.predict(X).ravel(), index=y.index)
    end = time.time()
    return y_pred, end-start


def train_predict(clf, X_train, y_train, X_test, y_test, results_df):
    results_df.at['Training time', len(y_train)] = train_classifier(clf, X_train, y_train)
    y_pred, t_pred = predict_labels(clf, X_train, y_train)
    results_df.at['F1 score (train)' , len(y_train)] = f1_score(y_train.values, y_pred>0.5)
    results_df.at['Precision (train)', len(y_train)] = precision_score(y_train.values, y_pred>0.5)
    results_df.at['Recall (train)'   , len(y_train)] = recall_score(y_train.values, y_pred>0.5)
    results_df.at['Accuracy (train)' , len(y_train)] = accuracy_score(y_train.values, y_pred>0.5)
    results_df.at['ROC AUC (train)'  , len(y_train)] = roc_auc_score(y_train.values, y_pred)
    y_pred, t_pred = predict_labels(clf, X_test, y_test)
    results_df.at['F1 score (test)' , len(y_train)] = f1_score(y_test.values, y_pred>0.5)
    results_df.at['Precision (test)', len(y_train)] = precision_score(y_test.values, y_pred>0.5)
    results_df.at['Recall (test)'   , len(y_train)] = recall_score(y_test.values, y_pred>0.5)
    results_df.at['Accuracy (test)' , len(y_train)] = accuracy_score(y_test.values, y_pred>0.5)
    results_df.at['ROC AUC (test)'  , len(y_train)] = roc_auc_score(y_test.values, y_pred)
    results_df.at['Prediction time' , len(y_train)] = t_pred
    return y_pred

def train_predict_new(clf, X_train, y_train, X_test, y_test, results_df, factor):
    results_df.at[factor, 'Training time'] = train_classifier(clf, X_train, y_train)
    y_pred, t_pred = predict_labels(clf, X_train, y_train)
    #results_df.at[factor, 'F1 score (train)'] = f1_score(y_train.values, y_pred>0.5)
    #results_df.at[factor, 'Precision (train)'] = precision_score(y_train.values, y_pred>0.5)
    #results_df.at[factor, 'Recall (train)'] = recall_score(y_train.values, y_pred>0.5)
    #results_df.at[factor, 'Accuracy (train)'] = accuracy_score(y_train.values, y_pred>0.5)
    #results_df.at[factor, 'ROC AUC (train)'] = roc_auc_score(y_train.values, y_pred)
    y_pred, t_pred = predict_labels(clf, X_test, y_test)
    results_df.at[factor, 'F1 score'] = f1_score(y_test.values, y_pred>0.5)
    results_df.at[factor, 'Precision'] = precision_score(y_test.values, y_pred>0.5)
    results_df.at[factor, 'Recall'] = recall_score(y_test.values, y_pred>0.5)
    results_df.at[factor, 'Accuracy'] = accuracy_score(y_test.values, y_pred>0.5)
    #results_df.at[factor, 'ROC AUC'] = roc_auc_score(y_test.values, y_pred)
    #results_df.at[factor, 'Prediction time'] = t_pred
    return y_pred
