import numpy as np
import torch
import warnings


from tqdm import tqdm




def _mAP_NP(database_hash, test_hash, database_labels, test_labels, args):  # R = 1000
    # binary the hash code
    R = args.R
    T = args.T
    database_hash = database_hash.astype(np.int32) * 2 - 1
    test_hash = test_hash.astype(np.int32) * 2 - 1

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)
    #data_dir = 'data/' + args.data_name
    #ids_10 = ids[:10, :]

    #np.save(data_dir + '/ids.npy', ids_10)
    APx = []
    Recall = []

    for i in tqdm(range(query_num), desc="mAP", leave=False, ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):  # for i=0
        label = test_labels[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / all_num.astype(float)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx


def mean_average_precision(database_hash: torch.FloatTensor, test_hash: torch.FloatTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor, K):
    AP_200 = list()
    AP_all = list()
    Precision_100 = list()
    Precision_200 = list()
    apPerClass = {}
    for i in range(400):
        apPerClass[i] = []
    vis = [0 for i in range(400)]
    # warnings.warn("mAP by torch is 1%% lower than numpy version.")
    count = 0
    ranklist_per_class = {}
    for queryX, queryLabels in tqdm(zip(test_hash, test_labels), desc='Caluate mAP', total= len(test_hash)):# tqdm(zip(test_hash, test_labels), leave=False, desc='MAP', ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):
        thisAP_200, thisAP_all, thisPrecision_100,thisPrecision_200, ids  = _partedMAP_all(database_hash, queryX[None, :], database_labels, queryLabels[None, :], K)
        AP_200.append(thisAP_200)
        AP_all.append(thisAP_all)
        if queryLabels.nonzero().item() == 390:
            ranklist_per_class[count] = ids[:10]
        # if vis[queryLabels.nonzero().item()] == 0:
        #     vis[queryLabels.nonzero().item()] = 1
        #     ranklist_per_class[count] = ids[:10]
        # idlist.append(ids[:10])
        apPerClass[queryLabels.nonzero().item()].append(thisAP_200.item())
        # apPerClass[queryLabels].append(thisAP_200)
        Precision_100.append(thisPrecision_100)
        Precision_200.append(thisPrecision_200)
        count = count + 1
    for i in apPerClass:
        if len(apPerClass[i])!=0:
            apPerClass[i] = np.mean(apPerClass[i])
    
    # # appen remaining AP
    # thisAP, thisR = _partedMAP(database_hash[batches * 64:], test_hash, database_labels[batches * 64:], test_labels, args)
    # AP.append(thisAP)
    # R.append(thisR)
   
    AP_200 = torch.cat(AP_200)
    AP_all = torch.cat(AP_all)
    Precision_100 = torch.cat(Precision_100)
    Precision_200 = torch.cat(Precision_200)
    return AP_200.mean(), AP_all.mean(), Precision_100.mean(),Precision_200.mean(), apPerClass, ranklist_per_class


def _partedMAP(database_hash: torch.FloatTensor, test_hash: torch.FloatTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor, K):  # R = 1000
    R = K

    # [1, Nb]
    sim = -((test_hash - database_hash) ** 2).sum(-1)

    # [Nq, R] top R queried from base
    _, ids = torch.topk(sim, R, -1, largest=True, sorted=True)

    # [Nq, R, nclass]
    queried_labels = database_labels[ids]
    # [Nq, R, nclass] [Nq, 1, nclass] -> [Nq, R] ordered matching result
    matched = (torch.logical_and(queried_labels, test_labels[:, None]).sum(-1) > 0).float()
    # [Nq] Does this query has any match?
    hasMatched = matched.sum(-1)
    # cum-sum along R dim
    L = torch.cumsum(matched, -1)
    # [Nq, R]
    P = L / torch.arange(1, R + 1, 1, device=L.device, dtype=torch.float)

    # [Nq] / [Nq]
    AP = (P * matched).sum(-1) / hasMatched
    # for results has no match, set to 0
    AP[hasMatched < 1] = 0

    # [Nq, Nb] -> [Nq], Recall base
    allRelevent = (torch.logical_and(test_labels[:, None], database_labels).sum(-1) > 0).sum(-1).float()

    # [Nq]
    Recall = hasMatched / allRelevent

    # [Nq]
    Precision = P[:, -1]

    # [Nq]
    return AP, Recall, Precision



def _partedMAP_all(database_hash: torch.FloatTensor, test_hash: torch.FloatTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor, K):  # R = 1000
    R = K

    # [1, Nb]
    sim = -((test_hash - database_hash) ** 2).sum(-1)
    sim = sim.reshape(-1)
    # [Nq, R] top R queried from base
    _, ids = torch.topk(sim, sim.shape[-1], -1, largest=True, sorted=True)
    
    # [Nq, R, nclass]
    queried_labels = database_labels[ids]
   
    # [Nq, R, nclass] [Nq, 1, nclass] -> [Nq, R] ordered matching result
    matched_100 = (torch.logical_and(queried_labels[:100], test_labels[:, None]).sum(-1) > 0).float()
    matched_200 = (torch.logical_and(queried_labels[:200], test_labels[:, None]).sum(-1) > 0).float()
    matched_all = (torch.logical_and(queried_labels, test_labels[:, None]).sum(-1) > 0).float()
    # [Nq] Does this query has any match?
    hasMatched_100 = matched_100.sum(-1)
    hasMatched_200 = matched_200.sum(-1)
    hasMatched_all = matched_all.sum(-1)
    # cum-sum along R dim
    L_100 = torch.cumsum(matched_100, -1)
    L_200 = torch.cumsum(matched_200, -1)
    L_all = torch.cumsum(matched_all, -1)
    # [Nq, R]
    P_100 = L_100 / torch.arange(1, 100 + 1, 1, device=L_100.device, dtype=torch.float)
    P_200 = L_200 / torch.arange(1, 200 + 1, 1, device=L_200.device, dtype=torch.float)
    P_all = L_all / torch.arange(1, sim.shape[-1] + 1, 1, device=L_all.device, dtype=torch.float)

    # [Nq] / [Nq]
    AP_100 = (P_100 * matched_100).sum(-1) / hasMatched_100
    AP_200 = (P_200 * matched_200).sum(-1) / hasMatched_200
    AP_all = (P_all * matched_all).sum(-1) / hasMatched_all
    # for results has no match, set to 0
    AP_100[hasMatched_100 < 1] = 0
    AP_200[hasMatched_200 < 1] = 0
    AP_all[hasMatched_all < 1] = 0

    # [Nq, Nb] -> [Nq], Recall base
    allRelevent = (torch.logical_and(test_labels[:, None], database_labels).sum(-1) > 0).sum(-1).float()

    # [Nq]
    Recall_100 = hasMatched_100 / allRelevent
    Recall_200 = hasMatched_200 / allRelevent
    Recall_all = hasMatched_all / allRelevent

    # [Nq]
    Precision_100 = P_100[:, -1]
    Precision_200 = P_200[:, -1]
    Precision_all = P_all[:, -1]

    # [Nq]
    return AP_200, AP_all, Precision_100, Precision_200, ids



@torch.no_grad()
def get_rank_list(database_hash: torch.BoolTensor, test_hash: torch.BoolTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor):
    precisions = list()
    recalls = list()
    pAtH2s = list()
    database_hash = database_hash.float() * 2 - 1
    for queryX, queryLabels in tqdm(zip(test_hash, test_labels), leave=False, desc='MAP', ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):
        # [1, Nb]
        precision, recall, pAtH2 = _partedRank(database_hash, queryX[None, :], database_labels, queryLabels[None, :])
        precisions.append(precision)
        recalls.append(recall)
        pAtH2s.append(pAtH2)
    # [Nb], [Nb], float
    return torch.cat(precisions).mean(0), torch.cat(recalls).mean(0), float(torch.tensor(pAtH2s).mean())


def _partedRank(database_hash: torch.FloatTensor, test_hash: torch.BoolTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor):  # R = all
    # [1, Nb]
    sim = (test_hash.float() * 2 - 1) @ database_hash.T

    bits = test_hash.shape[-1]
    h2 = bits - 4

    # [1, Nb] queried from base
    values, ids = torch.sort(sim, -1, descending=True)

    # the first index that distance > 2
    rankinsideH2 = torch.nonzero((values < h2)[0])[0]

    # [1, R, nclass]
    queried_labels = database_labels[ids]
    # [1, R, nclass] [1, 1, nclass] -> [1, R] ordered matching result
    matched = (torch.logical_and(queried_labels, test_labels[:, None]).sum(-1) > 0).float()
    cumsum = matched.cumsum(-1)
    # [1, R]
    precision = (cumsum / torch.arange(1, matched.shape[-1] + 1, 1, device=matched.device, dtype=torch.float))
    # [1, R]
    recall = (cumsum / matched.sum(-1, keepdim=True))



    # [1, R], [1, R], float
    return precision, recall, float(precision[0, rankinsideH2])
