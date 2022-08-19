from __future__ import print_function, division
import numpy as np
import sys
import os
import time
import shutil
import copy
import json
import pandas as pd
import ray
import itertools
from sklearn.model_selection import cross_val_score

pp_path = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(pp_path)
os.environ["PYTHONPATH"] = pp_path + ":" + os.environ.get("PYTHONPATH", "")
from clustering import *
from confident_learning import *
from utils import *
from neural_net import *
np.set_printoptions(precision=4)


###------###
NEED_MODEL_PARAMS_SEARCH = True # search for model training hyperparams

NEED_TRAINING = True            # train NN (first round)

NEED_MASK_COMPUTE = True        # compute masks

NEED_RETRAINING = True          # retrain NN on cleaned dataset (second round)

NEED_MASK_EVALUATION = True     # evaluate mask accuracy

NEED_MODEL_EVALUATION = True    # evaluate retrained model test accuracy

has_gpu = True

# 'Cardiotocography', 'CreditFraud', 'HAR', 'Letter', 'Mushroom', 'SatIm', 'SenDrive'
datasets = ['Cardiotocography', 'CreditFraud', 'HAR', 'Letter', 'Mushroom', 'SatIm', 'SenDrive']
seeds = list(range(10)) # list(range(10))

## train / retrain ##
batch_size = 1024
learning_rate = 1e-3        # Adam optimizer by default
weight_decay = 0
balance_class_weights = True
sampling_method = None      # method for upsampling minor classes. None, 'duplication', or 'SMOTE'
## train / retrain ##

## mask ##
methods = ['kmeans', 'no_clean', 'cl']
class_depend = True
ma_size = 5

# for CTRL clustering parameters search
cluster_params_options = {
    # for each n_clusters, 1 <= n_select_clusters <= n_clusters-1
    'n_clusters_options': [2,3],
    # for each n_windows, 1 <= window_thresh <= int(0.5*n_windows)+1
    'n_windows_options': [1,2,4],
}
mask_score_alpha = 0
## mask ##

## model evaluation ##
metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc_ovr']
## model evaluation ##

## model hyperparams search ##
cvn = 4     # number of CV folds
scoring = 'balanced_accuracy' # metric to decide the best training hyperparams set
scoring_mode = 'max'

# helper function for get_hiden_sizes_choices, returns a list of NN architectures
def get_nn_archis(
    num_hidden_layer_list,  # list. candidate number of hidden layers
    hidden_sizes_list,      # list of lists. each hidden layer has its candidate sizes
    ):
    num_hidden_layer_list.sort()
    hidden_sizes_allComb = []
    for num_hidden_layer in num_hidden_layer_list:
        if num_hidden_layer<1: continue
        if num_hidden_layer>len(hidden_sizes_list): break
        hidden_sizes_allComb += list(itertools.product(*(hidden_sizes_list[:num_hidden_layer])))
    return list(map(list, hidden_sizes_allComb))

# return list of canddidate NN hidden sizes
def get_hiden_sizes_choices(inp_size, dataset=None):
    # inp_size ~= #tr_sample/100
    if dataset in ['Cardiotocography', 'CreditFraud']:
        inp_size = inp_size/2
    if dataset in ['HAR', 'SatIm']:
        inp_size = inp_size*1.5
    if dataset in ['Letter', 'SenDrive']:
        inp_size = inp_size*10

    return get_nn_archis(
        num_hidden_layer_list=[2,3],                            # try 2- and 3- hidden-layer NNs
        hidden_sizes_list=[[int(i*inp_size) for i in [2,4]],    # candidate sizes of the 1st hidden layer
                           [int(i*inp_size) for i in [1,2]],    # candidate sizes of the 2nd hidden layer
                           [int(i*inp_size) for i in [1]]])     # candidate sizes of the 3rd hidden layer

# return list of canddidate training epochs
def get_epochs_choices(dataset):
    return [50, 100, 200, 300, 500, 800]
## model hyperparams search ##
###------###


data_template = 'datasets/{dataset}/seed_{seed}/data'
train_folder_template = 'training/{dataset}/seed_{seed}'
mask_template = 'mask/{dataset}/seed_{seed}/{method}/train_mask.npy'
model_template = 'retrained_model/{dataset}/seed_{seed}/{method}/model.pth.tar'
all_cl_methods = ['argmax', 'cl_pbc', 'conf_joint_only', 'cl_both', 'cl_pbnr']

# make clustering parameters search space
if NEED_MASK_COMPUTE:
    mask_params_search_space = []
    for n_clusters in cluster_params_options['n_clusters_options']:
        for n_select_clusters in range(1, n_clusters):
            for n_windows in cluster_params_options['n_windows_options']:
                for window_thresh in range(1, int(0.5*n_windows)+2):
                    mask_params_search_space.append({
                        'n_clusters': n_clusters,
                        'n_select_clusters': n_select_clusters,
                        'n_windows': n_windows,
                        'window_thresh': window_thresh,
                    })

# model training hyperparams to pass into the NN training function
model_params = {
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'num_fold': 1,
    
    'balance_class_weights': balance_class_weights,
    'sampling_method': sampling_method,
    
    'prune_percent': 0,
    'pruning_schedule': {},
}


def np_savefile(filepath, np_data):
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(filepath, np_data)



@ray.remote(num_gpus=(0.25 if has_gpu else 0), num_cpus=1)
def experiment(
    seed,
    dataset,
    model_params,
    NEED_MODEL_PARAMS_SEARCH=False,
    NEED_TRAINING=True,
    NEED_MASK_COMPUTE=True,
    NEED_RETRAINING=True,
    NEED_MASK_EVALUATION=True,
    NEED_MODEL_EVALUATION=True,
):
    # ----------- helper functions -----------
    def _train_helper(_modelpath, _record_train_loss, _mask=None):
        if _mask is None:
            # train with mask
            _mask = np.ones((num_samples,), dtype=int).astype(bool)
        clf = NN_sk(model_params, seed)
        clf.fit(X_train_arr[_mask], y_train_arr[_mask], _record_train_loss)
        
        if _record_train_loss:
            np_savefile(loss_filepath, clf.sample_train_loss)
        save_nn(_modelpath, clf.model)
    
    def _mask_helper():
        n_clusters = mask_params['n_clusters']
        n_select_clusters = mask_params['n_select_clusters']
        n_windows = mask_params['n_windows']
        window_thresh = mask_params['window_thresh']
        
        _mask = compute_mask(loss_mtx=loss_mtx,
                             loss_thresh=loss_thresh, ma_size=ma_size,
                             class_depend=class_depend, labels=y_train_arr,
                             n_clusters=n_clusters, n_select_clusters=n_select_clusters,
                             n_windows=n_windows, window_thresh=window_thresh,)
        
        mask_silhouette = get_silhouette_score(loss_mtx_ma, _mask.astype(int), class_depend=class_depend, labels=y_train_arr)
        mask_train_acc = model_evaluate(noisy_model, X_train_arr[_mask], y_train_arr[_mask], model_params['batch_size'], metrics=[scoring])[scoring]
        _, mask_loss_ratio = get_loss_ratio(loss_mtx_ma, _mask.astype(int), class_depend=False)
        _score = mask_silhouette * np.power(mask_train_acc*mask_loss_ratio, mask_score_alpha)
        return _mask, _score
    
    def _mask_evaluation_helper():
        tgt, pred = 1-gt_mask, 1-mask
        tmp_result = {
            'dataset': dataset,
            'seed': seed,
            'method': method,
            'accuracy': accuracy_score(tgt, pred),
            'balanced_accuracy': balanced_accuracy_score(tgt, pred),
            'f1': f1_score(tgt, pred),
            'f1_macro': f1_score(tgt, pred, average='macro'),
            'acc_pruned': sum(y_train_arr[mask] == y_train_clean_arr[mask]) / sum(mask.astype(int)),
            'remove_pct': 1 - np.sum(mask.astype(int) / len(mask))
        }
        return tmp_result
    
    def _retrain_evaluation_helper():
        model = load_nn(model_filepath, model_params)
        tmp_result = {
            'dataset': dataset,
            'seed': seed,
            'method': method,
        }
        tmp_result.update(model_evaluate(model, X_test_arr, y_test_arr, model_params['batch_size'], metrics))
        return tmp_result
    
    def _model_params_search_helper():
        clf = NN_sk(model_params, seed)
        tmp_result = {
            'dataset': dataset,
            'seed': seed,
            'hidden_sizes': hidden_sizes,
            'num_epoch': num_epoch,
        }
        skf = StratifiedKFold(n_splits=cvn, shuffle=True, random_state=seed)
        tmp_result.update({'score': 
            np.mean(cross_val_score(clf, X_train_arr, y_train_clean_arr, scoring=scoring, cv=skf))})
        return tmp_result
    # ----------- helper functions -----------
    
    data_folder = data_template.format(dataset=dataset, seed=seed)
    X_train_df, y_train_arr, X_test_df, y_test_arr, _, categorical_features, y_train_clean_arr = read_data(
        data_folder, load_y_train_clean=True)
    
    num_features = len(X_train_df.columns)
    num_samples = len(y_train_arr)
    num_class = len(np.unique(y_train_arr))
    output_size = num_class if num_class>2 else 1
    loss_thresh = 2*np.log(num_class)
    model_params.update({
        'feature_cols': list(X_train_df.columns),
        'categorical_features': categorical_features,
        'num_class': num_class,
        'output_size': output_size,
    })
    X_train_arr = X_train_df.to_numpy()
    X_test_arr = X_test_df.to_numpy()
    y_train_arr = np.squeeze(y_train_arr)               # noisy labels
    y_test_arr = np.squeeze(y_test_arr)
    y_train_clean_arr = np.squeeze(y_train_clean_arr)   # ground truth labels
    gt_mask = np.array(y_train_arr == y_train_clean_arr).astype(int)    # ground truth mask
    
    if NEED_MODEL_PARAMS_SEARCH:
        params_search_results = []
        for hidden_sizes in get_hiden_sizes_choices(num_features, dataset):
            for num_epoch in get_epochs_choices(dataset):
                model_params.update({
                    'hidden_sizes': hidden_sizes,
                    'num_epoch': num_epoch,
                })
                params_search_results.append(_model_params_search_helper())
    
    else:
        # load training hyperparams (list of hidden sizes and number of epochs)
        df_params = pd.read_csv('model_params.csv')
        model_params.update({
            'hidden_sizes': list(map(int, (df_params[dataset][0][1:-1]).split(','))),
            'num_epoch': int(df_params[dataset][1]),
        })
        
        mask_results = []
        evaluation_results = []
        
        folder_path = train_folder_template.format(dataset=dataset, seed=seed)
        loss_filepath = os.path.join(folder_path, 'train_loss.npy')
        noisy_model_filepath = os.path.join(folder_path, 'model.pth.tar')
        psx_filepath = os.path.join(folder_path, 'psx.npy')
        
        if NEED_TRAINING:
            _train_helper(noisy_model_filepath, True)
            
            if 'cl' in methods:  # for cl methods, train with cleanlab
                clf = NN_sk(model_params, seed)
                _, psx = cleanlab.latent_estimation.estimate_confident_joint_and_cv_pred_proba(
                    X=X_train_arr,
                    s=y_train_arr,
                    clf=clf,
                    seed=seed,
                )
                np_savefile(psx_filepath, psx)
        
        if NEED_MASK_COMPUTE:
            loss_mtx = np.load(loss_filepath)
            loss_mtx_ma = moving_avg(np.clip(loss_mtx, 0, loss_thresh), ma_size)
            noisy_model = load_nn(noisy_model_filepath, model_params)
            if 'cl' in methods:
                psx = np.load(psx_filepath)
        
        for method in methods:
            mask_filepath = mask_template.format(dataset=dataset, seed=seed, method=method)
            model_filepath = model_template.format(dataset=dataset, seed=seed, method=method)
            
            if NEED_MASK_COMPUTE:
                if method == 'no_clean':
                    mask = np.ones((num_samples,), dtype=int).astype(bool)
                
                elif method == 'cl':
                    # select the one with best mask accuracy
                    best_cl_method, best_cl_acc = 'cl_both', 0
                    for cl_method in all_cl_methods:
                        cl_mask_filepath = mask_template.format(dataset=dataset, seed=seed, method=cl_method)
                        cl_mask = compute_mask_cl(cl_method, y_train_arr, psx)
                        np_savefile(cl_mask_filepath, cl_mask)
                        
                        cur_acc = accuracy_score(gt_mask, cl_mask)
                        if cur_acc>best_cl_acc:
                            best_cl_acc = cur_acc
                            mask = cl_mask
                
                else: # CTRL
                    # select clustering parameters and its corresponding mask (by Eq.(1) in the paper)
                    best_mask_score = -np.inf
                    for mask_params in mask_params_search_space:
                        cur_mask, cur_score = _mask_helper()
                        if cur_score>best_mask_score:
                            mask = cur_mask
                            best_mask_score = cur_score
                
                np_savefile(mask_filepath, mask)
            
            if NEED_RETRAINING:
                mask = np.load(mask_filepath)
                _train_helper(model_filepath, False, mask)
            
            if NEED_MASK_EVALUATION:
                mask = np.load(mask_filepath)
                mask_results.append(_mask_evaluation_helper())
            
            if NEED_MODEL_EVALUATION:
                evaluation_results.append(_retrain_evaluation_helper())
    
    if NEED_MODEL_PARAMS_SEARCH:
        return params_search_results
    return mask_results, evaluation_results



end_time = time.time()
ray.init(include_dashboard=False, logging_level=0, log_to_driver=True, ignore_reinit_error=True)

# when model param search is True, will finish model training hyperparams search first
if NEED_MODEL_PARAMS_SEARCH:
    tests = []
    for seed in seeds:
        for dataset in datasets:
            tests.append(experiment.remote(
                seed,
                dataset,
                copy.deepcopy(model_params),
                NEED_MODEL_PARAMS_SEARCH=True,
            ))
    results = ray.get(tests)
    
    params_search_results = []
    for er in results:
        params_search_results += er
    pd.DataFrame(params_search_results).to_csv('model_params_search.csv', index=False)
    
    # select hyperparams sets with the best CV score and write them into a new file
    df = pd.read_csv('model_params_search.csv')
    df_params = {}
    for dataset in np.unique(df['dataset']):
        df_tmp = df[df['dataset']==dataset][['seed', 'hidden_sizes', 'num_epoch', 'score']]
        df_params[dataset] = ((1 if scoring_mode=='max' else -1) * df_tmp.groupby(
            ['hidden_sizes', 'num_epoch']).agg({'score': 'sum'})).idxmax().values[0]
    pd.DataFrame(df_params).to_csv('model_params.csv', index=False)


# model training &/ mask computation &/ evaluation
NEED_MODEL_PARAMS_SEARCH = False
if NEED_TRAINING or NEED_MASK_COMPUTE or NEED_RETRAINING or NEED_MASK_EVALUATION or NEED_MODEL_EVALUATION:
    tests = []
    for seed in seeds:
        for dataset in datasets:
            tests.append(experiment.remote(
                seed, 
                dataset,
                copy.deepcopy(model_params),
                NEED_MODEL_PARAMS_SEARCH,
                NEED_TRAINING,
                NEED_MASK_COMPUTE,
                NEED_RETRAINING,
                NEED_MASK_EVALUATION,
                NEED_MODEL_EVALUATION,
            ))
    results = ray.get(tests)
    
    mask_results = []
    evaluation_results = []
    for er in results:
        mask_results += er[0]
        evaluation_results += er[1]

    if NEED_MASK_EVALUATION:
        pd.DataFrame(mask_results).to_csv('mask_results.csv', index=False)
    if NEED_MODEL_EVALUATION:
        pd.DataFrame(evaluation_results).to_csv('evaluation.csv', index=False)
ray.shutdown()
print('\ntime: {:.1f} sec'.format(time.time()-end_time))


