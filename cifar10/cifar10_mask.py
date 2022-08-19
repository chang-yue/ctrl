from __future__ import print_function, division

import numpy as np
import sys
import os
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
import ray
import subprocess

pp_path = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(pp_path)
os.environ["PYTHONPATH"] = pp_path + ":" + os.environ.get("PYTHONPATH", "")
from clustering import *
from confident_learning import *
np.set_printoptions(precision=4)



###
dataset = 'cifar10'##
seeds = [0, 1, 2]
sparsities = [0, 20, 40, 60] #[0, 20, 40, 60]##
noise_rates = [0, 10, 20] #[0, 10, 20]##

NEED_MASK_COMPUTE = True  # for mask computation
methods = ['kmeans', 'cl', 'no_clean'] # 'kmeans', 'cl', 'no_clean'

class_depend = True
loss_thresh = 2*np.log(10)##
ma_size = 5
model_arch = 'resnet50'

NEED_PARAM_SEARCH = True  # for clustering parameters search
cluster_params_options = {
    # for each n_clusters, try 1 <= n_select_clusters <= n_clusters-1
    'n_clusters_options': [2,3],
    # for each n_windows, try 1 <= window_thresh <= int(0.5*n_windows)+1
    'n_windows_options': [1,2,4],
}
mask_score_alpha = 1##
###


folder_template = '{dataset}_noisy_labels__frac_zero_noise_rates__0_{sparsity}__noise_amount__0_{noise}/seed_{seed}'
s_template = '{dataset}_noisy_labels/{dataset}_noisy_labels__frac_zero_noise_rates__0_{sparsity}__noise_amount__0_{noise}/seed_{seed}/{dataset}_noisy_labels__frac_zero_noise_rates__0.{sparsity}__noise_amount__0.{noise}.json'
loss_template = '{dataset}_training{sm}_loss/{base_folder}/model_{model_arch}_train_loss{sf}.npy'
mask_template = '{dataset}_mask{scl}/{base_folder}/{method}/model_{model_arch}_train_mask.npy'
all_cl_methods = ['argmax', 'cl_pbc', 'conf_joint_only', 'cl_both', 'cl_pbnr']

# load clean (ground truth) labels
rfn = s_template.format(dataset=dataset, sparsity='00', noise='00', seed='0')
with open(rfn, 'r') as rf:
    d = json.load(rf)
y = np.asarray([v for k,v in d.items()])

# make clustering parameters search space
if NEED_PARAM_SEARCH:
    params_search_space = []
    for n_clusters in cluster_params_options['n_clusters_options']:
        for n_select_clusters in range(1, n_clusters):
            for n_windows in cluster_params_options['n_windows_options']:
                for window_thresh in range(1, int(0.5*n_windows)+2):
                    params_search_space.append({
                        'n_clusters': n_clusters,
                        'n_select_clusters': n_select_clusters,
                        'n_windows': n_windows,
                        'window_thresh': window_thresh,
                    })


def np_savefile(filepath, np_data):
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(filepath, np_data)



@ray.remote(num_gpus=(1 if NEED_PARAM_SEARCH else 0), num_cpus=2)
def experiment(
    seed,
    sparsity,
    noise,
    NEED_MASK_COMPUTE=False,
    NEED_PARAM_SEARCH=False,
):
    # ----------- helper functions -----------
    def _mask_compute(_params):
        n_clusters = _params['n_clusters']
        n_select_clusters = _params['n_select_clusters']
        n_windows = _params['n_windows']
        window_thresh = _params['window_thresh']
        
        mask = compute_mask(loss_mtx=loss_mtx,
                            loss_thresh=loss_thresh, ma_size=ma_size,
                            class_depend=class_depend, labels=s,
                            n_clusters=n_clusters, n_select_clusters=n_select_clusters,
                            n_windows=n_windows, window_thresh=window_thresh,)
        return n_clusters, n_select_clusters, n_windows, window_thresh, mask
    
    # get masked training accuracy
    evaluate_template = '\
    python {dataset}_train.py --evaluate \
    --arch {model_arch} --gpu 0 --workers 4 --batch-size 256 \
    --train-labels {dataset}_noisy_labels/{base_folder}/{dataset}_noisy_labels__frac_zero_noise_rates__0.{sparsity}__noise_amount__0.{noise}.json \
    --resume {model_path} \
    --dir-train-mask {mask_path} \
    datasets/datasets/{dataset}/{dataset}/ \
    '
    tmp_mask_template = '{dataset}_mask{scl}/{base_folder}/{method}/model_{model_arch}_train_mask_{param}.npy'
    def _get_mask_train_acc(ckpt):
        # temporarily save the mask file with the current clustering parameters
        tmp_mask_filepath = tmp_mask_template.format(
            dataset=dataset, scl='', base_folder=base_folder, method=method, model_arch=model_arch,
            param='{}_{}_{}_{}'.format(n_clusters, n_select_clusters, n_windows, window_thresh),
        )
        np_savefile(tmp_mask_filepath, mask)
        
        # calculate masked training accuracy
        model_path = '{}_training/{}/model_{}_{}.pth.tar'.format(dataset, base_folder, model_arch, ckpt)
        cmd = evaluate_template.format(dataset=dataset, sparsity=sparsity_str, noise=noise_str, model_arch=model_arch, base_folder=base_folder, model_path=model_path, mask_path=tmp_mask_filepath)
        result = subprocess.check_output(cmd, shell=True)
        acc1, _,acc5, _,acc_avg, _,acc_min = result.split(b"* Acc@1 ")[-1].strip().split()
        
        # remove temporary mask file
        cmd = 'rm {}'.format(tmp_mask_filepath)
        result = subprocess.check_output(cmd, shell=True)

        return float(acc1), float(acc_avg), float(acc_min)
    
    
    # compute scores
    def _result_helper(method_name):
        acc_pruned = sum(np.asarray(s)[mask] == y[mask]) / sum(mask.astype(int))
        tgt, pred = 1-gt_mask, 1-mask
        cur_result = {
            'seed': seed,
            'method': method_name,
            'accuracy': accuracy_score(tgt, pred),
            'balanced_accuracy': balanced_accuracy_score(tgt, pred),
            'f1': f1_score(tgt, pred),
            'f1_macro': f1_score(tgt, pred, average='macro'),
            #'confusion_matrix': confusion_matrix(tgt, pred),
            'acc_pruned': acc_pruned,
            'frac_zero_noise_rates': sparsity / 100.,
            'noise_amount': noise / 100.,
        }
        
        if NEED_PARAM_SEARCH:
            _loss_mtx = np.load(loss_filepath)
            _loss_mtx[np.where(_loss_mtx>loss_thresh)] = loss_thresh
            loss_mtx_ma = moving_avg(_loss_mtx, ma_size)
            
            mask_silhouette_score = get_silhouette_score(loss_mtx_ma, mask.astype(int), class_depend=class_depend, labels=s)
            mask_train_acc_best, mask_train_acc_avg_best, mask_train_acc_min_best = _get_mask_train_acc('best')
            mask_train_acc_last, mask_train_acc_avg_last, mask_train_acc_min_last = _get_mask_train_acc('_checkpoint')
            loss_ratio_avg, loss_ratio_last = get_loss_ratio(loss_mtx_ma, mask.astype(int), class_depend=False)
            keep_pct = sum(mask.astype(int)) / len(mask)
            
            cur_result.update({
                'n_clusters': n_clusters,
                'n_select_clusters': n_select_clusters,
                'n_windows': n_windows,
                'window_thresh': window_thresh,
                'mask_silhouette_score': mask_silhouette_score,
                'mask_train_acc_best': mask_train_acc_best,
                'mask_train_acc_avg_best': mask_train_acc_avg_best,
                'mask_train_acc_min_best': mask_train_acc_min_best,
                'mask_train_acc_last': mask_train_acc_last,
                'mask_train_acc_avg_last': mask_train_acc_avg_last,
                'mask_train_acc_min_last': mask_train_acc_min_last,
                'loss_ratio_avg': loss_ratio_avg,
                'loss_ratio_last': loss_ratio_last,
                'keep_pct': keep_pct,
                'comb_score': mask_silhouette_score * np.power(
                    (mask_train_acc_avg_last+mask_train_acc_avg_best)*loss_ratio_last, mask_score_alpha)
            })
        
        results.append(cur_result)
    # ----------- helper functions -----------
    
    results = []
    if noise<1e-2 and sparsity>0:
        return []
    sparsity_str = '{0:0=2d}'.format(sparsity)
    noise_str = '{0:0=2d}'.format(noise)
    
    if NEED_PARAM_SEARCH:
        # turn off mask computation/evaluation if in clustering parameters search mode
        NEED_MASK_COMPUTE = False
    else:
        df = pd.read_csv('mask_param_search.csv')
        mask_param_search = df[(df['seed']==seed) & (df['frac_zero_noise_rates']==sparsity/100.) & (df['noise_amount']==noise/100.)]
    
    s_filepath = s_template.format(dataset=dataset, sparsity=sparsity_str, noise=noise_str, seed=seed)
    with open(s_filepath, 'r') as rf:
        s = np.asarray(list(json.load(rf).values()))    # noisy labels
    gt_mask = np.array(s == y).astype(int)  # ground truth mask
    
    base_folder = folder_template.format(dataset=dataset, sparsity=sparsity_str, noise=noise_str, seed=seed)
    for method in methods:
        mask_filepath = mask_template.format(dataset=dataset, scl='', base_folder=base_folder, method=method, model_arch=model_arch)
        
        if NEED_MASK_COMPUTE or NEED_PARAM_SEARCH:
            if method == 'no_clean':
                if NEED_MASK_COMPUTE:
                    mask = np.ones((len(s),), dtype=int).astype(bool)
            
            # CTRL
            elif method != 'cl':
                loss_filepath = loss_template.format(dataset=dataset, base_folder=base_folder, model_arch=model_arch, sm='', sf='')
                loss_mtx = np.load(loss_filepath)
                
                if NEED_PARAM_SEARCH:
                    # compute the mask and clustering score of each (k, s, w, t) parameter set
                    for params in params_search_space:
                        n_clusters, n_select_clusters, n_windows, window_thresh, mask = _mask_compute(params)
                        _result_helper(method)
                
                else:
                    # choose the (k, s, w, t) pair with the highest clustering score
                    mask_param = mask_param_search[mask_param_search['method']==method]
                    idx = mask_param['comb_score'].idxmax()                    
                    n_clusters, n_select_clusters, n_windows, window_thresh, mask = _mask_compute(mask_param.loc[idx])
            
            # CL
            elif NEED_MASK_COMPUTE:
                # calculate masks using various cl methods and save the one with the best mask accuracy
                psx_filepath = '{}_mask_cl/{}/{}__train__model_{}__pyx.npy'.format(dataset, base_folder, dataset, model_arch)
                psx = np.load(psx_filepath)
                assert psx.shape == (len(s), len(np.unique(s, return_counts=False)))
                
                best_cl_method, best_cl_acc = 'cl_both', 0
                for cl_method in all_cl_methods:
                    cl_mask_filepath = mask_template.format(dataset=dataset, scl='_cl', base_folder=base_folder, method=cl_method, model_arch=model_arch)
                    cl_mask = compute_mask_cl(cl_method, s, psx)
                    np_savefile(cl_mask_filepath, cl_mask)
                    
                    cur_acc = accuracy_score(gt_mask, cl_mask)
                    if cur_acc>best_cl_acc:
                        best_cl_acc = cur_acc
                        best_cl_method = cl_method
                best_cl_mask_filepath = mask_template.format(dataset=dataset, scl='_cl', base_folder=base_folder, method=best_cl_method, model_arch=model_arch)
                mask = np.load(best_cl_mask_filepath)
            
            if NEED_MASK_COMPUTE:
                np_savefile(mask_filepath, mask)
        
        else:
            mask = np.load(mask_filepath)
        
        if not NEED_PARAM_SEARCH:
            _result_helper(method)
    
    return results



# ray.init(include_dashboard=False, logging_level=0, log_to_driver=False, ignore_reinit_error=True)
ray.init()

# helper function for tasks distributation on Ray
def ray_helper():
    ray_tests = []
    for seed in seeds:
        for sparsity in sparsities:
            for noise in noise_rates:
                ray_tests.append(experiment.remote(
                    seed,
                    sparsity,
                    noise,
                    NEED_MASK_COMPUTE,
                    NEED_PARAM_SEARCH,
                ))
    ray_results = ray.get(ray_tests)

    results = []
    for rr in ray_results:
        results += rr
    return pd.DataFrame(results)


if NEED_PARAM_SEARCH:
    df = ray_helper()
    df.to_csv('mask_param_search.csv', index=False)
    NEED_PARAM_SEARCH = False

if NEED_MASK_COMPUTE:
    df = ray_helper()
    df.to_csv('mask_results.csv', index=False)

ray.shutdown()


