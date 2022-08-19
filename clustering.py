import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def compute_mask(
    loss_mtx,           # loss matrix of shape |samples|x|epochs|
    
    loss_thresh,        # loss threshold
    ma_size,            # moving average window size

    class_depend,       # whether run each class separately or not
    labels,             # if class_depend is True, input the (potentially noisy) labels; else None

    n_clusters,             # number of clusters, k
    n_select_clusters,      # number of selected clusters, s
    
    n_windows,              # number of windows (if divide the loss matrix uniformly), w
    window_thresh,          # window threshold, t
    windows=None,           # if do not wish to divide the loss matrix uniformly, can specify
                            #     the windows as a list of (start_epoch, end_epoch) tuples
    ):
    
    loss_mtx[np.where(loss_mtx>loss_thresh)] = loss_thresh  # clamp
    loss_mtx = moving_avg(loss_mtx, ma_size)                # smooth

    # divide the loss matrix vertically into windows of equal sizes if do not use customized windows
    if windows is None:
        windows = []
        interval = len(loss_mtx[0]) / max(n_windows,1)
        for i in range(n_windows):
            windows.append((int(i*interval), int((i+1)*interval)))

    # separate each row/sample of the loss matrix based on its (noisy) label
    # if class-independent, then treat all samples as if they are from the same class
    _s = labels if class_depend else np.zeros((len(loss_mtx),), dtype=int)
    class_ids, class_counts = np.unique(_s, return_counts=True)

    # initialize window mask votes and mask
    mask_votes = np.ones((len(loss_mtx), len(windows)), dtype=int)
    mask = np.zeros((len(loss_mtx),), dtype=int)

    # go through each window and each class
    for w, (start, end) in enumerate(windows):
        for class_id,class_count in zip(class_ids, class_counts):
            if class_count<2:
                continue
            
            # extract the relavent loss matrix fraction
            cur_loss_mtx = loss_mtx[np.where(_s==class_id)[0], start:end]

            # run KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cur_loss_mtx)

            # sort cluster centers by area under loss curve
            argsort = np.argsort(np.sum(kmeans.cluster_centers_, axis=1))[::-1]

            # assign labels that belong to the top s clusters noisy, else clean
            for k in range(n_select_clusters):
                mask_votes[np.arange(len(_s))[_s==class_id][kmeans.labels_==argsort[k]], w] = 0

    # sum up votes along windows and apply the window threshold
    mask[np.sum(mask_votes, axis=1)>=window_thresh] = 1
    return mask.astype(bool)


# Calculate moving average of the matrix along rows
def moving_avg(mtx, ma_size):
    cumsum = np.cumsum(mtx, axis=1)
    ma = (cumsum[:,ma_size:] - cumsum[:,:-ma_size]) / ma_size
    # fill values to make the returned matrix the same shape as the input matrix
    return np.concatenate([ma[:,:1]*np.ones((1,ma_size)), ma], axis=1)



# Compute Silhouette score of loss clustering assignment, based on mask
# If class-dependent, need to provide the noisy labels as well
def get_silhouette_score(loss_mtx, mask, class_depend=False, labels=None):
    if not class_depend:
        if len(np.unique(mask))<2:
            return 0
        return silhouette_score(loss_mtx, mask)
    
    class_scores = []
    class_ids, class_counts = np.unique(labels, return_counts=True)
    for class_id,class_count in zip(class_ids, class_counts):
        if class_count<2:
            continue
        tmp = np.where(labels==class_id)[0]
        if len(np.unique(mask[tmp]))>=2:
            class_scores.append(silhouette_score(loss_mtx[tmp,:], mask[tmp]))
    return np.mean(class_scores)


# Compute (average and last) loss ratios between masked clean and masked noisy samples
# If class-dependent, need to provide the noisy labels as well
def get_loss_ratio(loss_mtx, mask, class_depend=False, labels=None):
    if not class_depend:
        labels = np.zeros((len(mask),), dtype=int)
    
    avg, last = [], []
    class_ids, class_counts = np.unique(labels, return_counts=True)
    for class_id,class_count in zip(class_ids, class_counts):
        loss_mtx_0 = loss_mtx[np.logical_and(mask==0, labels==class_id)]
        loss_mtx_1 = loss_mtx[np.logical_and(mask==1, labels==class_id)]
        avg.append(np.mean(loss_mtx_0) / (np.mean(loss_mtx_1)) + 1e-10)
        last.append(np.mean(loss_mtx_0[:,-1]) / (np.mean(loss_mtx_1[:,-1])) + 1e-10)
    return np.mean(avg), np.mean(last)


