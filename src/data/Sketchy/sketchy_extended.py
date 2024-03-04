import os
import pickle
import numpy as np
import glob


def create_trvalte_splits(args):
    _BASE_PATH = os.path.join(args.code_path, "src/data/")
    path_sk = os.path.join(args.dataset_path, 'Sketchy', 'sketch')
    path_im = os.path.join(args.dataset_path, 'Sketchy', 'extended_photo')

    tr_classes, va_classes, te_classes, cid_mask, splits_sk = trvalte_per_domain(args, path_sk, is_eccv_split=bool(args.is_eccv_split))
    _, _, _, _, splits_im = trvalte_per_domain(args, path_im, is_eccv_split=bool(args.is_eccv_split))

    with open(os.path.join(_BASE_PATH, 'Sketchy', 'glove300'+'.pkl'), 'rb') as f:
        semantic_vec = pickle.load(f)

    splits = {}
    splits['query_tr'] = np.array(splits_sk['tr'])
    splits['gallery_tr'] = np.array(splits_im['tr'])
    splits['tr'] = np.array(splits_sk['tr'] + splits_im['tr'])
    splits['query_va'] = np.array(splits_sk['va'])
    splits['gallery_va'] = np.array(splits_im['va'])
    splits['query_te'] = np.array(splits_sk['te'])
    splits['gallery_te'] = np.array(splits_im['te'])

    print('\n# Classes - Tr:{}; Va:{}; Te:{}'.format(len(tr_classes), len(va_classes), len(te_classes)))

    # print('#Tr Sketches:{}, #Tr Images:{}'.format(len(splits_sk['tr']), len(splits_im['tr'])))
    # print('#Te Sketches:{}, #Te Images:{}'.format(len(splits_sk['te']), len(splits_im['te'])))

    return {'tr_classes':tr_classes, 'va_classes':va_classes, 'te_classes':te_classes, 'semantic_vec':semantic_vec, 'cid_mask':cid_mask, 
            'splits':splits}

def create_trvalte_splits_frequency(args):
    _BASE_PATH = os.path.join(args.code_path, "src/data/")
    path_sk = os.path.join(args.root_path, 'Sketchy', 'sketch')
    path_im = os.path.join(args.root_path, 'Sketchy', 'extended_photo')

    tr_classes, va_classes, te_classes, cid_mask, splits_sk = trvalte_per_domain(args, path_sk, is_eccv_split=bool(args.is_eccv_split))
    _, _, _, _, splits_im = trvalte_per_domain(args, path_im, is_eccv_split=bool(args.is_eccv_split))

    with open(os.path.join(_BASE_PATH, 'Sketchy', 'glove300'+'.pkl'), 'rb') as f:
        semantic_vec = pickle.load(f)

    splits = {}
    splits['tr'] = np.array(splits_im['tr'])
    splits['te_dg'] = np.array(splits_sk['tr'])
    splits['te_zsl'] = np.array(splits_im['te'])
    

    print('\n# Classes - Tr:{}; Va:{}; Te:{}'.format(len(tr_classes), len(va_classes), len(te_classes)))

    # print('#Tr Sketches:{}, #Tr Images:{}'.format(len(splits_sk['tr']), len(splits_im['tr'])))
    # print('#Te Sketches:{}, #Te Images:{}'.format(len(splits_sk['te']), len(splits_im['te'])))

    return {'tr_classes':tr_classes, 'va_classes':va_classes, 'te_classes':te_classes, 'semantic_vec':semantic_vec, 'cid_mask':cid_mask, 
            'splits':splits}


def trvalte_per_domain(args, datapath, is_eccv_split=True):
    _BASE_PATH = os.path.join(args.code_path, "src/data/")
    all_fls = np.array([f for f in glob.glob(os.path.join(datapath ,'*/*.*'))])
    all_clss = np.array([f.split('/')[-2] for f in all_fls])

    classes = sorted(os.listdir(datapath))

    if is_eccv_split:
        with open(os.path.join(_BASE_PATH, 'Sketchy/test_classes_eccv_2018.txt'), 'r') as fp:
        # with open('test_classes_eccv_2018.txt', 'r') as fp:
            te_classes = fp.read().splitlines()

        with open(os.path.join(_BASE_PATH, 'Sketchy/cid_mask_eccv_split.pkl'), 'rb') as f:
            cid_mask = pickle.load(f)
    else:
        with open(os.path.join(_BASE_PATH, 'Sketchy/test_classes_random_split.txt'), 'r') as fp:
            te_classes = fp.read().splitlines()

        with open(os.path.join(_BASE_PATH, 'Sketchy/cid_mask_random_split.pkl'), 'rb') as f:
            cid_mask = pickle.load(f)

    np.random.seed(0)
    trval_classes = np.setdiff1d(classes, te_classes).tolist()
    tr_classes = np.random.choice(trval_classes, int(0.9*len(trval_classes)), replace=False).tolist()
    va_classes = np.setdiff1d(trval_classes, tr_classes).tolist()
    

    
    tr_classes = trval_classes

    fls_tr = []
    fls_va = []
    fls_te = []

    for c in te_classes:
        fls_te += all_fls[np.where(all_clss==c)[0]].tolist()
    
    for c in tr_classes:
        fls_tr += all_fls[np.where(all_clss==c)[0]].tolist()

    for c in va_classes:
        fls_va += all_fls[np.where(all_clss==c)[0]].tolist()

    splits = {}
    splits['tr'] = fls_tr
    splits['va'] = fls_va
    splits['te'] = fls_te

    return tr_classes, va_classes, te_classes, cid_mask, splits