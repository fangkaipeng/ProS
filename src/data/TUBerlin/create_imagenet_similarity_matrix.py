import os
import re
import numpy as np
import glob
import collections
from nltk.corpus import wordnet as wn
import pickle


def trvalte_per_domain(datapath):

    all_fls = np.array([f for f in glob.glob(os.path.join(datapath, '*/*.*'))])
    all_clss = np.array([f.split('/')[-2] for f in all_fls])

    classes = sorted(os.listdir(datapath))

    with open('test_classes_random_split.txt', 'r') as fp:
        te_classes = fp.read().splitlines()
    
    with open('val_classes.txt', 'r') as fp:
        va_classes = fp.read().splitlines()

    tr_classes = np.setdiff1d(classes, np.union1d(te_classes, va_classes))

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

    return tr_classes, va_classes, te_classes, splits


verbal = True
similarity_thrh = 0.2
wnid_pattern = re.compile("^n\d{8}_[0-9]+")

root_path = '/media/mvp18/Study/ZS-SBIR/datasets/'
path_im = os.path.join(root_path, 'TUBerlin', 'images')
tr_classes, va_classes, te_classes, splits_im = trvalte_per_domain(path_im)

wnid_dict=dict()
for file in splits_im['tr']+splits_im['va']:
    cname = file.split('/')[-2]
    fname = file.split('/')[-1]
    if wnid_pattern.match(fname.split('.')[0]):
        wnid = fname.split('_')[0]
        if cname not in wnid_dict.keys():
            wnid_dict[cname]=wnid
        else:
            if not wnid in wnid_dict[cname]:
                # print('new wnid')
                # print(cname, wnid, wnid_dict[cname])
                wnid_dict[cname] = ','.join([wnid_dict[cname], wnid])

for cname in tr_classes.tolist() + va_classes:
	if cname not in wnid_dict.keys():
	    # print('new category with no wnid')
	    # print(cname, filename)
	    wnid_dict[cname]='???'
            
if verbal:
    print('no wnid found for:')
    for cname in wnid_dict.keys():
        if '???' in wnid_dict[cname]:
            print(cname)
    
    print('more than one wnid found for:')
    for cname in wnid_dict.keys():
        if ',' in wnid_dict[cname]:
            print(wnid_dict[cname])


synset_dict = dict()
for cname in wnid_dict.keys():
    if wnid_dict[cname]=='???':
        if cname == 'present':
            synset_dict[cname] = [wn.synsets(cname)[1]]
        elif cname == 'teddy-bear':
            synset_dict[cname] = [wn.synsets('teddy')[0]]
        elif cname == 'flying saucer':
            synset_dict[cname] = [wn.synsets('ufo')[0]]
        elif cname == 'potted plant':
            synset_dict[cname] = [wn.synsets('pot_plant')[0]]
        elif cname == 'santa claus':
            synset_dict[cname] = [wn.synsets('santa_claus')[0]]
        elif cname in ['person walking', 'person sitting']:
            synset_dict[cname] = [wn.synsets('person')[0]]
        elif cname == 'fire hydrant':
            synset_dict[cname] = [wn.synsets('fire_hydrant')[0]]
        elif cname == 'outlet':
            synset_dict[cname] = [wn.synsets('wall_socket')[0]]
        elif cname == 'flower with stem':
            synset_dict[cname] = [wn.synsets('flower')[0]]
        elif cname == 'human-skeleton':
            synset_dict[cname] = [wn.synsets('skeleton')[2]]
        elif len(wn.synsets(cname))==0:
            synset_dict[cname] = None
        else:
            synset_dict[cname] = [wn.synsets(cname)[0]]
    elif ',' in wnid_dict[cname]:
        synset_dict[cname] = []
        for wnid_i in wnid_dict[cname].split(','):
            synset_dict[cname].append(wn.synset_from_pos_and_offset('n', int(wnid_i[1:])))
    else:
        synset_dict[cname] = [wn.synset_from_pos_and_offset('n', int(wnid_dict[cname][1:]))]
        
        
# read in imagenet index and synsets
imagenet_dict_file = '../imagenet_label_to_wordnet_synset.txt'
with open(imagenet_dict_file, 'r') as fh:
    file_content = fh.readlines()
    
imagenet_dict = dict()
for li in range(0,len(file_content),3):
    wnid = file_content[li].strip().split('\'')[-2]
    imagenet_dict[li//3] = wn.synset_from_pos_and_offset('n', int(wnid.split('-')[0]))
    

# make correspondance matrix from sketchy categories to imagenet classes
wn_matrix = dict()
hypo = lambda s: s.hyponyms()
for cname in synset_dict.keys():
    wn_matrix[cname] = np.zeros(len(imagenet_dict))
    if synset_dict[cname] is None:
        continue
        
    for ss in synset_dict[cname]:
        for ik,iss in imagenet_dict.items():
            if ss == iss:
                wn_matrix[cname][ik] = 1
            elif iss in list(ss.closure(hypo)) or ss in list(iss.closure(hypo)):
#             elif iss in ss.hyponyms():
                wn_matrix[cname][ik] = 1
            elif ss.path_similarity(iss) > similarity_thrh:
                wn_matrix[cname][ik] = ss.path_similarity(iss)
                # wn_matrix[cname][ik] = 1
                
one_to_one=0
for cname in synset_dict.keys():
    if synset_dict[cname] is None:
        continue
        
    for ss in synset_dict[cname]:
        if ss in imagenet_dict.values():
            one_to_one += 1
            
print('number of exact matches:')
print(one_to_one)

hist_ls = []
for cname in list(wn_matrix.keys()):
    hist_ls.append(np.sum(wn_matrix[cname]))
    
cname_most = list(wn_matrix.keys())[np.where(np.array(hist_ls)==max(hist_ls))[0][0]]
print('the category that has the most corresponding classes is: {}, {}'.format(synset_dict[cname_most], max(hist_ls)))
iilist = np.where(wn_matrix[cname_most]==1)[0]
for ii in iilist:
    print(file_content[ii*3+1].strip())
    
cnter = collections.Counter(hist_ls)
total_c = 0
for ck,cv in cnter.items():
    total_c += int(cv)*int(ck)
    print('{} category corresponds to {} classes in imagenet'.format(int(cv),int(ck)))
    
print(total_c)

wn_matrix_np = np.zeros((len(wn_matrix),1000))
for ci,cname in enumerate(list(wn_matrix.keys())):
    wn_matrix_np[ci] = wn_matrix[cname]

print('Number of ImageNet classes that are not matched:')
print(np.sum(np.sum(wn_matrix_np, axis=0) == 0))

to_save = os.path.join('cid_mask_random_split.pkl')
with open(to_save, 'wb') as fh:
    pickle.dump(wn_matrix, fh)