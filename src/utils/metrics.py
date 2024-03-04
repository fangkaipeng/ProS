import sys
import time
import numpy as np
import torch.nn.functional as F

import multiprocessing
from joblib import Parallel, delayed

# metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

from utils.mAP import mean_average_precision



def prec(actual, predicted, k):

    act_set = set(actual)

    if k is not None:
        pred_set = set(predicted[:k])
        pr = len(act_set & pred_set) / min(k, len(pred_set))
    else:
        pred_set = set(predicted)
        pr = len(act_set & pred_set) / max(len(pred_set), 1)

    return pr


def rec(actual, predicted, k):

    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / max(len(act_set), 1)

    return re


def precak(sim, str_sim, k=None):

    act_lists = [np.nonzero(s)[0] for s in str_sim]
    pred_lists = np.argsort(-sim, axis=1)
    # num_cores = min(multiprocessing.cpu_count(), 2)
    num_cores = multiprocessing.cpu_count()
    nq = len(act_lists)
    preck = Parallel(n_jobs=num_cores)(delayed(prec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    reck = Parallel(n_jobs=num_cores)(delayed(rec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    # preck = [prec(act_lists[iq], pred_lists[iq], k) for iq in range(nq)]
    # reck = [rec(act_lists[iq], pred_lists[iq], k) for iq in range(nq)]

    return np.mean(preck), np.mean(reck)


def aps(sim, str_sim):
    nq = str_sim.shape[0]
    # num_cores = min(multiprocessing.cpu_count(), 2)
    num_cores = multiprocessing.cpu_count()
    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))
    # aps = [average_precision_score(str_sim[iq], sim[iq]) for iq in range(nq)]
    return aps


def apsak(sim, str_sim, k=None):
    idx = (-sim).argsort()[:, :k]
    sim_k = np.array([sim[i, id] for i, id in enumerate(idx)])
    str_sim_k = np.array([str_sim[i, id] for i, id in enumerate(idx)])
    idx_nz = np.where(str_sim_k.sum(axis=1) != 0)[0]
    sim_k = sim_k[idx_nz]
    str_sim_k = str_sim_k[idx_nz]
    aps_ = np.zeros((sim.shape[0]), dtype=np.float)
    aps_[idx_nz] = aps(sim_k, str_sim_k)
    return aps_


def compute_retrieval_metrics(query, query_class, database, database_class):
    # query = query / query.norm(dim=-1, keepdim=True)
    # database = database / database.norm(dim=-1, keepdim=True)
    query_class = F.one_hot(query_class, num_classes=1000) > 0 # 2400, 1000
    database_class = F.one_hot(database_class, num_classes=1000) > 0 # 2400, 1000
    ap200,ap_all,prec_100,prec200,apPerclass, ranklist_per_class = mean_average_precision(database, query, database_class, query_class, 200)
    
    return {'mAP@200': ap200, 'prec@200': prec200,'mAP@all':ap_all,'prec@100':prec_100, 'class_ap':apPerclass, 'ranklist':ranklist_per_class}
