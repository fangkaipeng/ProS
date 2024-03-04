import copy
import os
import numpy as np
from random import random
from PIL import Image, ImageOps
# import cv2
import torch
import torch.utils.data as data
import torchvision

from scipy.spatial.distance import cdist
import torchvision.transforms as transforms
from src.data import _BASE_PATH


class BaselineDataset(data.Dataset):
    def __init__(self, fls, transforms=None):
        
        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample, clss, sample_domain

    def __len__(self):
        return len(self.fls)

class BaselineDataset_path(data.Dataset):
    def __init__(self, fls, transforms=None):
        
        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample, clss, sample_domain, str(self.fls[item])

    def __len__(self):
        return len(self.fls)


class CuMixloader(data.Dataset):
    
    def __init__(self, fls, clss, doms, dict_domain, transforms=None):
        
        self.fls = fls
        self.clss = clss
        self.domains = doms
        self.dict_domain = dict_domain
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample, clss, sample_domain

    def __len__(self):
        return len(self.fls)


class PairedContrastiveImageDataset(data.Dataset):
    def __init__(self, fls, clss, doms, dict_domain, dict_clss, transforms, nviews, nothers):
        self.fls = fls
        self.clss = clss
        self.cls_ids = self.convert_text_to_number(self.clss, dict_clss)
        self.domains = doms
        self.dict_domain = dict_domain
        self.dict_clss = dict_clss
        self.transforms = transforms
        self.idx = torch.arange(len(self.cls_ids))
        self.nviews = nviews
        self.nothers = nothers

    def convert_text_to_number(self, clss, dict_clss):
        return torch.tensor([dict_clss[i] for i in clss])

    def __getitem__(self, item):
        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        clss_name = self.clss[item]
        clss = self.cls_ids[item]
        
        allSamples = [sample]

        for _ in range(self.nothers):
            allSamples.append(self.pickRandomSample(clss)[0])

        allImages = list()
        for i in allSamples:
            for _ in range(self.nviews):
                allImages.append(self.transforms(i))
        return torch.stack(allImages), clss_name, sample_domain

    def pickRandomSample(self, clss):
        targetIdx = self.idx[self.cls_ids == clss]
        randIdx = targetIdx[torch.randperm(len(targetIdx))[0]]

        sample_domain = self.domains[randIdx]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[randIdx])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[randIdx]).convert(mode='RGB')

        return sample, self.dict_domain[self.domains[randIdx]], self.cls_ids[randIdx]

    def __len__(self):
        return len(self.fls)


class PairedContrastiveImageDataset_SameDomain(data.Dataset):
    def __init__(self, fls, clss, doms, dict_domain, dict_clss, transforms, nviews, nothers):
        self.fls = fls
        self.clss = clss
        self.cls_ids = self.convert_text_to_number(self.clss, dict_clss)
        self.domains = doms
        self.dom_ids = self.convert_text_to_number(self.domains, dict_domain)
        self.dict_domain = dict_domain
        self.dict_clss = dict_clss
        self.transforms = transforms
        self.idx = torch.arange(len(self.cls_ids))
        self.nviews = nviews
        self.nothers = nothers

    def convert_text_to_number(self, clss, dict_clss):
        return torch.tensor([dict_clss[i] for i in clss])

    def __getitem__(self, item):
        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')

        clss = self.cls_ids[item]

        allSamples = [sample]

        for _ in range(self.nothers):
            allSamples.append(self.pickRandomSample(clss, self.dom_ids[item])[0])

        allImages = list()
        for i in allSamples:
            for _ in range(self.nviews):
                allImages.append(self.transforms(i))
        return torch.stack(allImages), clss

    def pickRandomSample(self, clss, dom):
        targetIdx = self.idx[torch.logical_and(self.cls_ids == clss, self.dom_ids == dom)]
        randIdx = targetIdx[torch.randperm(len(targetIdx))[0]]

        sample_domain = self.domains[randIdx]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[randIdx])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[randIdx]).convert(mode='RGB')

        return sample, self.dict_domain[self.domains[randIdx]], self.cls_ids[randIdx]

    def __len__(self):
        return len(self.fls)


class SAKELoader(data.Dataset):
    def __init__(self, fls, cid_mask, transforms=None):
        
        self.fls = fls
        self.cid_mask = cid_mask
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample, clss, self.cid_mask[clss]

    def __len__(self):
        return len(self.fls)


class SAKELoader_with_domainlabel(data.Dataset):
    def __init__(self, fls, cid_mask=None, transforms=None):
        
        self.fls = fls
        self.cid_mask = cid_mask
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
            domain_label = np.array([0])
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
            domain_label = np.array([1])
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.cid_mask is not None:
            return sample, clss, self.cid_mask[clss], domain_label
        else:
            return sample, clss, domain_label

    def __len__(self):
        return len(self.fls)


class Doodle2Search_Loader(data.Dataset):
    def __init__(self, fls_sketch, fls_image, semantic_vec, tr_classes, dict_clss, transforms=None):

        self.fls_sketch = fls_sketch
        self.fls_image = fls_image
        
        self.cls_sketch = np.array([f.split('/')[-2] for f in self.fls_sketch])
        self.cls_image = np.array([f.split('/')[-2] for f in self.fls_image])

        self.tr_classes = tr_classes
        self.dict_clss = dict_clss
        
        self.semantic_vec = semantic_vec
        # self.sim_matrix = np.exp(-np.square(cdist(self.semantic_vec, self.semantic_vec, 'euclidean'))/0.1)
        cls_euc = cdist(self.semantic_vec, self.semantic_vec, 'euclidean')
        cls_euc_scaled = cls_euc/np.expand_dims(np.max(cls_euc, axis=1), axis=1)
        self.sim_matrix = np.exp(-cls_euc_scaled)

        self.transforms = transforms
        

    def __getitem__(self, item):

        sketch = ImageOps.invert(Image.open(self.fls_sketch[item])).convert(mode='RGB')
        sketch_cls = self.cls_sketch[item]
        sketch_cls_numeric = self.dict_clss.get(sketch_cls)
        
        w2v = torch.FloatTensor(self.semantic_vec[sketch_cls_numeric, :])

        # Find negative sample
        possible_classes = self.tr_classes[self.tr_classes!=sketch_cls]
        sim = self.sim_matrix[sketch_cls_numeric, :]
        sim = np.array([sim[self.dict_clss.get(x)] for x in possible_classes])
        
        # norm = np.linalg.norm(sim, ord=1) # Similarity to probability
        # sim = sim/norm
        sim /= np.sum(sim)
        
        image_neg_cls = np.random.choice(possible_classes, 1, p=sim)[0]
        image_neg = Image.open(np.random.choice(self.fls_image[np.where(self.cls_image==image_neg_cls)[0]], 1)[0]).convert(mode='RGB')

        image_pos = Image.open(np.random.choice(self.fls_image[np.where(self.cls_image==sketch_cls)[0]], 1)[0]).convert(mode='RGB')

        if self.transforms is not None:
            sketch = self.transforms(sketch)
            image_pos = self.transforms(image_pos)
            image_neg = self.transforms(image_neg)

        return sketch, image_pos, image_neg, w2v


    def __len__(self):
        return len(self.fls_sketch)


class JigsawDataset(data.Dataset):
    def __init__(self, fls, transforms=None, jig_classes=30, bias_whole_image=0.9):
        
        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])

        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image # biases the training procedure to show the whole image more often
    
        self._image_transformer = transforms['image']
        self._augment_tile = transforms['tile']
        
        def make_grid(x):
            return torchvision.utils.make_grid(x, self.grid_size, padding=0)
        self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile
    
    def get_image(self, item):
        
        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        return self._image_transformer(sample)
        
    def __getitem__(self, item):
        
        img = self.get_image(item)
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1) # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
            
        data = torch.stack(data, 0)
        data = self.returnFunc(data)
        
        return torch.cat([self._augment_tile(img), data], 0), order, self.clss[item]

    def __len__(self):
        return len(self.fls)

    def __retrieve_permutations(self, classes):
        all_perm = np.load(os.path.join(_BASE_PATH, 'permutations_%d.npy' % (classes)))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm