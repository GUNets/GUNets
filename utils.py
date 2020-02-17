import sys
import time
import json
import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp



def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = 1/rowsum
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_transductive(dataset_str, root_dir='', return_label=True, modified=False, semi=False):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    if modified:
        names[-1] = 'graph_lite'
    objects = []
    for i in range(len(names)):
        with open(root_dir+"data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:

            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    if semi:
        idx_train = range(len(y))
    else:
        idx_train = range(len(ally)-500)
    idx_val = range(len(ally)-500, len(ally))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    if return_label:
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,labels
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def loadRedditFromNPZ(dataset_dir, root_dir='', modified=False):
    if modified:
        adj = sp.load_npz(root_dir + 'data/' + dataset_dir + "_adj_lite.npz")
    else:
        adj = sp.load_npz(root_dir+'data/' + dataset_dir+"_adj.npz")
    data = np.load(root_dir+'data/' + dataset_dir+".npz")

    return (adj, data['feats'], data['y_train'], data['y_val'],
            data['y_test'], data['train_index'], data['val_index'],
            data['test_index'])

def load_data(dataset, semi=False, modified=False):
    if dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,labels = load_transductive(
            dataset, return_label=True, modified=modified,semi=semi)
        tr_idx = list(np.where(train_mask)[0])
        va_idx = list(np.where(val_mask)[0])
        ts_idx = list(np.where(test_mask)[0])
        features = features.toarray()

    elif dataset in ['reddit', 'reddit_raw']:
        adj, features, y_train, y_val, y_test, tr_idx, va_idx, ts_idx = loadRedditFromNPZ(dataset, modified=modified)
        if not modified:
            adj = adj + adj.T
        labels = np.zeros(adj.shape[0])
        labels[tr_idx] = y_train
        labels[va_idx] = y_val
        labels[ts_idx] = y_test
        labels = one_hot(labels.astype(np.int))
    else:
        raise ValueError("dataset string is not allowable")
    return adj, features, labels, tr_idx, va_idx, ts_idx


def load_big(path='../data/', prefix='ppi'):
    adj_full = sp.load_npz(path + './{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = sp.load_npz(path + './{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open(path + './{}/role.json'.format(prefix)))
    feats = np.load(path + './{}/feats.npy'.format(prefix))
    class_map = json.load(open(path + './{}/class_map.json'.format(prefix)))
    if isinstance(class_map['0'], int):
        labels = np.zeros((feats.shape[0], 1))
    else:
        labels = np.zeros((feats.shape[0], len(class_map['0'])))
    assert len(class_map) == feats.shape[0]
    for _ in class_map.keys():
        labels[int(_)] = class_map[_]

    return adj_full, feats, labels, role['tr'], role['va'], role['te']


def one_hot(values):
    n_values = np.max(values) + 1
    one_hot = np.eye(n_values)[values]
    return one_hot




