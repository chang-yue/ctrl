from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import numpy as np
import pandas as pd
import pickle
import os
np.set_printoptions(precision=4)



# --------------------------------------------------------------------
# data processing

import collections
class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


def shuffle_split_data(df, train_fraction, seed=0):
    df_train, df_test = train_test_split(df, train_size=train_fraction, random_state=seed, stratify=df["Target"])
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    return df_train, df_test

def shuffle_data(df, seed=0):
    df_ = shuffle(df, random_state=seed)
    df_.reset_index(inplace=True, drop=True)
    return df_


# plot histograms of the training data
def plot_hist(df, figsize, nrow, ncol):
    plt.figure(figsize=figsize)
    for i, c in enumerate(df.columns):
        plt.subplot(nrow,ncol,i+1)
        sns.histplot(df[c])
        plt.title(c)
        plt.xlabel('')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# scaling
def scale_features(df_train, df_test, cols_list, scaler_list):
    df_train_scaled = pd.DataFrame(columns=df_train.columns)
    df_test_scaled = pd.DataFrame(columns=df_test.columns)
    
    for cols,scaler in zip(cols_list, scaler_list):
        if len(cols)==0: continue
        cols_comp = list(OrderedSet(df_train.columns) - OrderedSet(cols))
        
        df_train_cols = df_train[cols]
        scaler.fit(df_train_cols)
        df_train_scaled[cols_comp] = df_train[cols_comp] if len(df_train_scaled)==0 else df_train_scaled[cols_comp]
        df_train_scaled[cols] = pd.DataFrame(scaler.transform(df_train_cols), columns=cols)
        
        df_test_scaled[cols_comp] = df_test[cols_comp] if len(df_test_scaled)==0 else df_test_scaled[cols_comp]
        df_test_scaled[cols] = pd.DataFrame(scaler.transform(df_test[cols]), columns=cols)

    if len(cols_list)==0:
        df_train_scaled, df_test_scaled = df_train, df_test
    
    return df_train_scaled, df_test_scaled


def save_data(folderpath, df_train, df_test, con_features, cat_features, y_train_clean_arr=None):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    df_train.to_csv(folderpath+'/train.csv', index=False, header=True)
    df_test.to_csv(folderpath+'/test.csv', index=False, header=True)

    with open(folderpath+'/continuous_features.txt', 'wb') as fp:
        pickle.dump(con_features, fp)

    with open(folderpath+'/categorical_features.txt', 'wb') as fp:
        pickle.dump(cat_features, fp)

    if y_train_clean_arr is not None:
        np.save(folderpath+'/y_train_clean.npy', y_train_clean_arr)


def read_data(folderpath, load_y_train_clean=False):
    df_train = pd.read_csv(folderpath+'/train.csv')
    df_test = pd.read_csv(folderpath+'/test.csv')

    X_train_df, y_train = df_train.drop("Target", axis=1), df_train["Target"]
    X_test_df, y_test = df_test.drop("Target", axis=1), df_test["Target"]

    with open(folderpath+'/continuous_features.txt', 'rb') as fp:
        continuous_features = pickle.load(fp)

    with open(folderpath+'/categorical_features.txt', 'rb') as fp:
        categorical_features = pickle.load(fp)

    y_train_arr = np.expand_dims(y_train, axis=-1)
    y_test_arr = np.expand_dims(y_test, axis=-1)
    y_train_clean_arr = np.load(folderpath+'/y_train_clean.npy') if load_y_train_clean else None
    y_train_clean_arr = np.expand_dims(y_train_clean_arr, axis=-1)
    return X_train_df, y_train_arr, X_test_df, y_test_arr, continuous_features, categorical_features, y_train_clean_arr



# --------------------------------------------------------------------
# MISC

def get_sampling_fn(sampling_method=None, categorical_features=None, feature_cols=None, random_state=0):
    if sampling_method=='duplication': 
        sampling_fn = RandomOverSampler(sampling_strategy='minority', random_state=random_state)
    elif sampling_method=='SMOTE':
        cat_feature_indices = []
        if categorical_features is not None and feature_cols is not None:
            cat_feature_index = [feature_cols.index(col) for col in categorical_features if col in feature_cols]
        if len(cat_feature_indices)==0: # all continuous
            sampling_fn = SMOTE(sampling_strategy='minority', random_state=random_state)
        elif len(cat_feature_indices)==len(feature_cols): # all categorical
            return get_sampling_fn('duplication')
        else:
            sampling_fn = SMOTENC(categorical_features=cat_feature_indices, sampling_strategy='minority', random_state=random_state)
    else:
        sampling_fn = None
    return sampling_fn


# return {metric: score} dictionary
def get_stats(pred, y, metrics, pred_proba=None):
    results = {}
    for m in metrics:
        if m=='accuracy':
            results.update({'accuracy': accuracy_score(y, pred)})
        elif m=='balanced_accuracy':
            results.update({'balanced_accuracy': balanced_accuracy_score(y, pred)})
        elif m=='f1_macro':
            results.update({'f1_macro': f1_score(y, pred, average='macro')})
        elif m=='roc_auc_ovr':
            results.update({'roc_auc_ovr': 
                            roc_auc_score(y, pred_proba if pred_proba is not None else pred, 
                                average='macro', multi_class='ovr')})
        elif m=='confu_mtx':
            results.update({'confu_mtx': confusion_matrix(y, pred)})
    return results


def save_output(filepath, model, save_model_fn, params=None):
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
            os.makedirs(folder)
    
    save_model_fn(filepath, model)

    if params is not None:
        with open(folderpath+'/params.pkl', 'wb') as fp:
            pickle.dump(params, fp)




