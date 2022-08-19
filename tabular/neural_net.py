import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.nn.utils.prune as prune
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import BaseEstimator
import ray
import os
import time
from utils import *
np.set_printoptions(precision=4)


# --------------------------------------------------------------------
# NN

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(m.bias)

class NN(nn.Module):
    def __init__(self,
        input_size,     # int
        output_size,    # int
        hidden_sizes,   # list of int
        ):
        super(NN, self).__init__()

        sizes = [input_size] + hidden_sizes + [output_size]
        self.output_dim = output_size

        # build NN
        self.layers = nn.ModuleList()
        parameters_to_prune = []
        for i in range(len(sizes)-1):
            l = nn.Linear(sizes[i], sizes[i+1])
            self.layers.append(l)
            parameters_to_prune.append((l, 'weight'))
        
        self.apply(weights_init)  # weight initilization
        self.parameters_to_prune = tuple(parameters_to_prune)

    # ReLU-activated
    def forward(self, x):
        out = x
        for i in range(len(self.layers)-1):
            out = nn.ReLU()(self.layers[i](out))
        return self.layers[-1](out)



# --------------------------------------------------------------------
# training

def get_dataLoader(X, y, isTrain, batch_size=1024):
    drop_last = False
    if isTrain and len(y)>batch_size and (len(y)%batch_size)<batch_size/3: drop_last = True
    data = torch.utils.data.TensorDataset(torch.from_numpy(X).type(torch.float), torch.from_numpy(y).type(torch.long))
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=isTrain, pin_memory=True, 
            drop_last=drop_last, sampler=None)  # must shuffle/sample because of drop_last, and sampling fn sorts by target


# return {metric: score} dictionary using the model
def nn_get_stats(dataLoader, model, device, metrics=['accuracy', 'f1_macro']):
    model.eval()
    output, target = [], []
    with torch.no_grad():
        for i, (X, y) in enumerate(dataLoader, 0):
            X = Variable(X).to(device)
            output.append(model(X))
            target.append(y)

    pred_proba = np.concatenate([
        torch.sigmoid(o).cpu().numpy() if model.output_dim==1 else
        F.softmax(o, dim=1).cpu().numpy()
        for o in output])
    target = np.concatenate(tuple(target))
    if model.output_dim==1: pred = np.round(pred_proba)
    else: pred = np.argmax(pred_proba, axis=-1)
    
    pred_proba = pred_proba if ('roc_auc_ovr' in metrics and len(np.unique(y))>2) else None
    return get_stats(pred, target, metrics, pred_proba)


# append sample losses at the current epoch to sample_loss_container
def record_loss_fn(model, containers, device):
    model.eval()
    loader_track = containers['loader_track']
    criterion = containers['criterion']
    sample_loss_container = containers['sample_loss_container']
    
    losses = []
    with torch.no_grad():
        for i, (X, y) in enumerate(loader_track):
            X, y = Variable(X).to(device), Variable(y).to(device)
            pred = model(X)

            if model.output_dim>1:
                temp = criterion(pred, torch.squeeze(y))
                if device!='cpu':
                    temp = temp.cpu().numpy()
                temp = np.expand_dims(temp, axis=-1)
            else:
                temp = criterion(pred, y.float())
                if device!='cpu':
                    temp = temp.cpu().numpy()
            losses.append(temp)

    sample_loss_container.append(np.concatenate(tuple(losses)))


def iterative_pruning(model, epoch, prune_percent, pruning_schedule):
    if epoch in pruning_schedule:
        if pruning_schedule[epoch]=='prune':
            prune_layers(model, prune_percent)
        elif pruning_schedule[epoch]=='remove':
            remove_prune(model)

def prune_layers(model, prune_percent):
    for module in model.layers:
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_percent/100.)
    # prune.global_unstructured(model.parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=prune_percent/100.,)

def remove_prune(model):
    for module in model.layers:
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            try: prune.remove(module, 'weight')
            except: pass
            try: prune.remove(module, 'bias')
            except: pass


def train_NN(
    X_train_arr,
    y_train_arr,
    params,
    seed=0,
    record_train_loss=False,
    cv_seed=0,
    train_verbose=-1,
    ):
    
    ## helper functions
    def _get_sampled_data(X, y):
        if sampling_fn is not None: 
            X_samp, y_samp = sampling_fn.fit_resample(X, y)
            y_samp = np.expand_dims(y_samp, axis=-1)
        else: 
            X_samp, y_samp = X, y
        return X_samp, y_samp # Note: need shuffling after sampling
    
    def _model_init():
        _model = NN(input_dim, output_dim, hidden_sizes)
        ###if torch.cuda.device_count()>1: _model = nn.DataParallel(_model)
        _model = _model.to(device)
        _optimizer = optim.Adam(_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if output_dim>1:
            _criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            pos_weight = None if class_weights is None else class_weights[1]/class_weights[0]
            _criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return _model, _optimizer, _criterion
    
    # adjust num_epoch by prune percent
    def _get_max_num_epoch():
        # return int(num_epoch * (100/(100-prune_percent))**2)
        return int(num_epoch)

    def _train():
        model.train()
        for i, (X, y) in enumerate(train_dataLoader, 0):
            X, y = Variable(X).to(device), Variable(y).to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, torch.squeeze(y) if model.output_dim>1 else y.float())
            loss.backward()
            optimizer.step()
    ## end of helper functions


    # parse parameters
    hidden_sizes = params['hidden_sizes']
    num_class = params['num_class']
    num_fold = params['num_fold']

    prune_percent = params['prune_percent']
    pruning_schedule = params['pruning_schedule']
    
    batch_size = params['batch_size']
    num_epoch = params['num_epoch']
    
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    
    balance_class_weights = params['balance_class_weights']
    sampling_method = params['sampling_method']
    feature_cols = params['feature_cols']
    categorical_features = params['categorical_features']

    # for key in params:
    #     exec(key+"=params['"+key+"']")  # exec is incompatible with Ray
    
    # setup
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    input_dim = X_train_arr.shape[-1]
    output_dim = num_class if num_class>2 else 1

    # balance class or resample samples by class
    class_weights = None
    if balance_class_weights:
        class_weights = torch.Tensor(len(y_train_arr) / num_class / (np.bincount(y_train_arr, minlength=num_class)+0.1)).to(device)
    sampling_fn = get_sampling_fn(sampling_method, categorical_features, feature_cols)
    
    # prepare data into (folds of) train and/or val
    # NOTE: if require memory efficiency, only store indices
    y_train_arr = np.expand_dims(y_train_arr, axis=-1)
    if num_fold == 1:
        X_train_samp, y_train_samp = _get_sampled_data(X_train_arr, y_train_arr)
        train_folds = [get_dataLoader(X_train_samp, y_train_samp, True, batch_size)]
    
    else:
        train_folds = []
        kf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=cv_seed)
        for fold, (train_index, val_index) in enumerate(kf.split(range(X_train_arr.shape[0]),y_train_arr)):
            X_train_fold_samp, y_train_fold_samp = _get_sampled_data(X_train_arr[train_index], y_train_arr[train_index])
            train_folds.append(get_dataLoader(X_train_fold_samp, y_train_fold_samp, True, batch_size))
    
    # adjust training settings
    max_num_epoch = _get_max_num_epoch()
    
    # setup sample training loss recording
    if record_train_loss:
        train_loader_track = get_dataLoader(X_train_arr, y_train_arr, False, batch_size)
        sample_train_loss = []
        train_loss_container = {
            'loader_track': train_loader_track,
            'criterion': nn.CrossEntropyLoss(reduction='none') if output_dim>1 else 
                         nn.BCEWithLogitsLoss(reduction='none'),
            'sample_loss_container': sample_train_loss,
        }
    else:
        train_loss_container = None
    
    # train each (train, val) pair
    for fold in range(num_fold):
        train_dataLoader = train_folds[fold]
        
        model, optimizer, criterion = _model_init()
        
        for epoch in range(max_num_epoch):
            _train()
            iterative_pruning(model, epoch+1, prune_percent, pruning_schedule)
            if train_loss_container is not None:
                record_loss_fn(model, train_loss_container, device)
        remove_prune(model)

    return model, np.array(sample_train_loss)[...,0].T if record_train_loss else None



# --------------------------------------------------------------------
# scikit-learn-compatible NN

class NN_sk(BaseEstimator):
    def __init__(self, params, seed):
        self.params = params
        self.seed = seed
    
    def fit(self, X, y, record_train_loss=False, sample_weight=None):
        self.model, self.sample_train_loss = train_NN(
            X, y,
            self.params,
            self.seed,
            record_train_loss=record_train_loss,
        )
        
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    
    def predict_proba(self, X):
        device = next(self.model.parameters()).device
        self.model.eval()
        X = torch.from_numpy(X).type(torch.float)
        output = []
        with torch.no_grad():
            X = Variable(X).to(device)
            output.append(self.model(X))
        
        pred_proba = np.concatenate([
            torch.sigmoid(o).cpu().numpy() if self.model.output_dim==1 else
            F.softmax(o, dim=1).cpu().numpy()
            for o in output])
        
        if self.model.output_dim==1:
            return np.hstack([1-pred_proba, pred_proba])
        return pred_proba
    
    def score(self, X, y, sample_weight=None):
        pass



# --------------------------------------------------------------------
# MISC

def model_evaluate(model, X, y, batch_size, metrics=['accuracy', 'f1_macro']):
    device = next(model.parameters()).device
    dataLoader = get_dataLoader(X, y, False, batch_size)
    return nn_get_stats(dataLoader, model, device, metrics)


def save_nn(filepath, model):
    model_folder = os.path.dirname(filepath)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), filepath)


def load_nn(filepath, params):
    model = NN(len(params['feature_cols']), params['output_size'], params['hidden_sizes'])
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model


