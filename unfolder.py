import time
import random
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

NUM_NODES, FEATURES, ADJ_TRAIN, NUM_FEATURES, MAX_SAMP_NEI = 0, 0, 0, 0, 0
IF_BAGGING, SAMP_PRE_NUM, SAMP_NUM, SAMP_TIMES, DEGREE = 0, 0, 0, 0, 0
SORTED, WEIGHT, TR_SET  = 0, 0, 0
RAW_ADJ, ADJ_SUM, MAX_DEGREE = 0, 0, 0
N_JOBS = 0


def set_seed(seed):
    print('Unfolder Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = 1 / rowsum
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def compute_adj_element(l):
    adj_map = NUM_NODES + np.zeros((l[1] - l[0], MAX_DEGREE), dtype=np.int)
    sub_adj = RAW_ADJ[l[0]: l[1]]
    for v in range(l[0], l[1]):
        neighbors = np.nonzero(sub_adj[v - l[0], :])[1]
        if v in TR_SET:
            neighbors = np.array(list(set(neighbors).intersection(TR_SET)))
        len_neighbors = len(neighbors)
        if len_neighbors > MAX_DEGREE:
            if SORTED:
                weight_sort = np.argsort(-ADJ_SUM[neighbors])
                neighbors = neighbors[weight_sort[:MAX_DEGREE]]
            else:
                neighbors = np.random.choice(neighbors, MAX_DEGREE, replace=False)
            adj_map[v - l[0]] = neighbors
        else:
            adj_map[v - l[0], :len_neighbors] = neighbors
    return adj_map


def compute_adjlist_parallel(sp_adj, max_degree, batch=50):
    global RAW_ADJ, MAX_DEGREE
    RAW_ADJ = sp_adj
    MAX_DEGREE = max_degree
    index_list = []
    for ind in range(0, NUM_NODES, batch):
        index_list.append([ind, min(ind + batch, NUM_NODES)])
    with Pool(N_JOBS) as pool:
        adj_list = pool.map(compute_adj_element, index_list)
    adj_list.append(NUM_NODES + np.zeros((1, MAX_DEGREE), dtype=np.int))
    adj_map = np.vstack(adj_list)
    return adj_map


def init(adj, feature_data, tr_set, samp_pre_num, samp_num, samp_times, degree=2, max_degree=32,
         max_samp_nei=3, if_normalized=False, degree_normalized=False, if_self_loop=True, if_bagging=True, if_sort=True,
         weight='same', n_jobs=2, seed=42):
    """ Init the shared data for multiprocessing
        Max_neighbor denotes the max allowed sampling neighbor
        number for a single node in traj sampling.
        Preventing the large degree neighbors have higher effects.
    """
    t = time.time()
    global NUM_NODES, FEATURES, ADJ_TRAIN, NUM_FEATURES, MAX_SAMP_NEI, WEIGHT
    global IF_BAGGING, SAMP_PRE_NUM, SAMP_NUM, SAMP_TIMES, N_JOBS, DEGREE
    global SORTED, ADJ_SUM, TR_SET
    if if_self_loop:
        adj = adj + sp.eye(adj.shape[0])
    set_seed(seed)
    NUM_FEATURES = feature_data.shape[1]
    NUM_NODES = feature_data.shape[0]
    TR_SET = tr_set
    SAMP_PRE_NUM = samp_pre_num
    SAMP_NUM = samp_num
    SAMP_TIMES = samp_times
    DEGREE = degree
    MAX_SAMP_NEI = max_samp_nei
    IF_BAGGING = if_bagging
    N_JOBS = n_jobs
    WEIGHT = weight
    if degree_normalized:
        adj_sum = np.sum(adj, axis=-1)
        feature_data = feature_data / np.array(adj_sum)
    null_feature = np.zeros((1, NUM_FEATURES))
    feature_data = np.vstack([feature_data, null_feature])

    if if_normalized:
        # feature_data = row_normalize(feature_data)
        # print('Normalized in %.2f s' % (time.time() - t))

        scaler = StandardScaler()
        scaler.fit(feature_data)
        feature_data = scaler.transform(feature_data)
    FEATURES = feature_data
    if if_sort:
        SORTED = True
        ADJ_SUM = np.array(np.sum(adj, axis=1)).reshape([-1])
    else: SORTED = False
    if NUM_NODES > 10000:
        ADJ_TRAIN = compute_adjlist_parallel(adj, max_degree)
    else:
        ADJ_TRAIN = compute_adjlist(adj, max_degree)
    if not IF_BAGGING:
        SAMP_TIMES = 1
    print('Init in %.2f s' % (time.time() - t))


# def get_traj_child(parent, sample_num=0):
#     '''
#     If sample_num == 0 return all the neighbors
#     '''
#
#     traj_list = []
#     for p in parent:
#         neigh = np.unique(ADJ_TRAIN[p].reshape([-1]))
#         if len(neigh) > 1:
#             neigh = neigh[neigh != NUM_NODES]
#         neigh = np.random.choice(neigh, min(MAX_SAMP_NEI, len(neigh)), replace=False)
#         t_array = np.hstack(
#             [p * np.ones((len(neigh), 1)).astype(np.int), neigh.reshape([-1, 1])])
#         traj_list.append(t_array)
#     traj_array = np.unique(np.vstack(traj_list), axis=0)
#     if traj_array.shape[0] > 1:
#         traj_array = traj_array[traj_array[:, -1] != NUM_NODES]
#     if sample_num:
#         traj_array = traj_array[
#             np.random.choice(
#                 traj_array.shape[0], min(sample_num, traj_array.shape[0]), replace=False)]
#     return traj_array

def get_traj_child(parent, sample_num=0):
    '''
    If sample_num == 0 return all the neighbors
    '''

    traj_list = []
    for p in parent:
        if type(p) == np.ndarray and p.shape[0]>1:
            neigh = np.unique(ADJ_TRAIN[p[-1]])
        else:
            neigh = np.unique(ADJ_TRAIN[p])
        if len(neigh) > 1:
            neigh = neigh[neigh != NUM_NODES]
        t_array = np.hstack(
            [p * np.ones((len(neigh), 1)).astype(np.int), neigh.reshape([-1, 1])])
        traj_list.append(t_array)
    traj_array = np.unique(np.vstack(traj_list), axis=0)
    if traj_array.shape[0] > 1:
        traj_array = traj_array[traj_array[:, -1] != NUM_NODES]
    if sample_num:
        traj_array = traj_array[
            np.random.choice(
                traj_array.shape[0], min(sample_num, traj_array.shape[0]), replace=False)]
    return traj_array


# def get_bgun_traj(idx):
#     '''
#     Get the trajectory set of a given node under the bagging gun setting.
#     '''
#     traj_list = [np.array(idx), []]
#     s1_nei = np.unique(ADJ_TRAIN[idx])
#     if len(s1_nei) > 1:
#         s1_nei = s1_nei[s1_nei != NUM_NODES]
#     for _ in range(DEGREE - 2):
#         s1_nei = get_traj_child(s1_nei, 0)
#     traj_list[1] = [
#         get_traj_child(
#             s1_nei[np.random.choice(list(range(len(s1_nei))), min(len(s1_nei), SAMP_PRE_NUM), replace=False)], SAMP_NUM)
#         for _ in range(SAMP_TIMES)]
#     return traj_list


def get_gun_traj(idx):
    '''
    Get the trajectory set of a given node under the naive gun setting.
    '''
    traj_list = [np.array(idx), []]
    whole_trajs = np.unique(ADJ_TRAIN[idx])
    for _ in range(DEGREE - 1):
        whole_trajs = get_traj_child(whole_trajs, 0)
    traj_list[1] = [whole_trajs]
    return traj_list


def get_gun_emb(idx):
    '''
    Generate the gun trajectory embedding of an appointed node
    including:
        1. get gun trajectory
        2. get mean embedding of each trajectory
    '''
    traj = get_gun_traj(idx)
    # traj[1] = [filter_traj(traj[1][0])]
    emb = get_traj_emb(traj)
    return emb

def get_traj_emb(traj):
    '''
    Get and stack the embedding of a given trajset
    '''
    emb_center = FEATURES[traj[0]]
    if WEIGHT == 'rw':
        emb_traj = np.stack(
            list(map(lambda x: np.mean(FEATURES[x].reshape([-1, DEGREE * NUM_FEATURES]), axis=0), traj[1]))).reshape(
            [DEGREE, -1])
    elif WEIGHT == 'same':
        unique_traj = [
            [np.unique(_[:, __]) for __ in range(_.shape[1])]
            for _ in traj[1]]
        emb_traj = np.mean(
            [np.concatenate([np.mean(FEATURES[__], axis=0) for __ in _])
             for _ in unique_traj], axis=0).reshape(DEGREE, -1)
    else:
        raise ('Weight Paremeter %s not defined' % WEIGHT)

    emb = np.vstack(
        [emb_center,
         emb_traj])
    return emb

def get_emb_multi(idx_list):
    '''
    Generate the gun trajectory embedding of an appointed node
    including:
        1. get gun trajectory
        2. get mean embedding of each trajectory
    '''

    emb_list = []
    for idx in idx_list:
        emb = get_gun_emb(idx)
        emb_list.append(emb)
    return np.stack(emb_list)


def get_batch_emb(node_list, batch=100, if_bagging=None, if_print=True):
    t = time.time()
    global IF_BAGGING, SAMP_TIMES
    IF_BAGGING_BAK = IF_BAGGING
    SAMP_TIMES_BAK = SAMP_TIMES
    if if_bagging is not None:
        IF_BAGGING = if_bagging
    if not IF_BAGGING:
        SAMP_TIMES = 1
    global RAW_ADJ, MAX_DEGREE
    batch_node_list = []
    for ind in range(0, len(node_list), batch):
        batch_node = node_list[ind: min(ind + batch, len(node_list))]
        batch_node_list.append(batch_node)
    with Pool(N_JOBS) as pool:
        emb_list = pool.map(get_emb_multi, batch_node_list)
    if if_print:
        print('Get embedding in %.2f second' % (time.time() - t))
    IF_BAGGING = IF_BAGGING_BAK
    SAMP_TIMES = SAMP_TIMES_BAK
    return np.vstack(emb_list)
#