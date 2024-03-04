import numpy as np
import random
import torch
from torch.utils.data import sampler


# Here we define a Sampler that has all the samples of each batch from the same domain,
# same as before but not distributed

# gives equal number of samples per domain, ordered across domains

class BalancedSampler(sampler.Sampler):

    def __init__(self, domain_ids, samples_per_domain, domains_per_batch=5, iters='min'):
        
        self.n_doms = domains_per_batch
        self.domain_ids = domain_ids
        random.seed(0)
        
        self.dict_domains = {}
        self.indeces = {}

        for i in range(self.n_doms):
            self.dict_domains[i] = []
            self.indeces[i] = 0

        self.dpb = domains_per_batch
        self.dbs = samples_per_domain
        self.bs = self.dpb*self.dbs

        for idx, d in enumerate(self.domain_ids):
            self.dict_domains[d].append(idx)

        min_dom = 10000000
        max_dom = 0

        for d in self.domain_ids:
            if len(self.dict_domains[d]) < min_dom:
                min_dom = len(self.dict_domains[d])
            if len(self.dict_domains[d]) > max_dom:
                max_dom = len(self.dict_domains[d])

        if iters == 'min':
            self.iters = min_dom // self.dbs
        elif iters == 'max':
            self.iters = max_dom // self.dbs
        else:
            self.iters = int(iters)

        for idx in range(self.n_doms):
            random.shuffle(self.dict_domains[idx])
    

    def _sampling(self, d_idx, n):
        if self.indeces[d_idx] + n >= len(self.dict_domains[d_idx]):
            self.dict_domains[d_idx] += self.dict_domains[d_idx]
        self.indeces[d_idx] = self.indeces[d_idx] + n
        return self.dict_domains[d_idx][self.indeces[d_idx] - n:self.indeces[d_idx]]

    
    def _shuffle(self):
        sIdx = []
        for i in range(self.iters):
            for j in range(self.n_doms):
                sIdx += self._sampling(j, self.dbs)
        return np.array(sIdx)

    
    def __iter__(self):
        return iter(self._shuffle())

    
    def __len__(self):
        return self.iters * self.bs