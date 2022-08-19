# --------------------------------------------------------------------
# Confident Learning (CL) methods
# (We used cleanlab v1.0 for our experiments.)

# https://github.com/cleanlab/cleanlab
# Northcutt, C.; Jiang, L.; and Chuang, I. 2021. 
# Confident learning: Estimating uncertainty in dataset labels. 
# Journal of Artificial Intelligence Research.


import cleanlab
import numpy as np

def compute_mask_cl(
    cl_method,  # cl method name
    s,          # (potentially noisy) labels
    psx,        # model predictions
    ):

    if cl_method == 'conf_joint_only':
        label_error_indices = cleanlab.latent_estimation.compute_confident_joint(
            s, psx, return_indices_of_off_diagonals=True)[1]
        label_error_mask = np.zeros(len(s), dtype=bool)
        for idx in label_error_indices:
            label_error_mask[idx] = True
        mask = ~label_error_mask

    elif cl_method == 'argmax':
        mask = ~cleanlab.baseline_methods.baseline_argmax(psx, s)

    elif cl_method == 'cl_pbc':
        mask = ~cleanlab.pruning.get_noise_indices(s, psx, prune_method='prune_by_class', n_jobs=1)

    elif cl_method == 'cl_pbnr':
        mask = ~cleanlab.pruning.get_noise_indices(s, psx, prune_method='prune_by_noise_rate', n_jobs=1)

    elif cl_method == 'cl_both':
        mask = ~cleanlab.pruning.get_noise_indices(s, psx, prune_method='both', n_jobs=1)

    else:
        mask = np.ones(s.shape).astype(bool)

    return mask

