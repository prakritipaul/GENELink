import pandas as pd
import torch
from torch.utils.data import Dataset
import random as rd
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import torch.nn as nn

## Understood ##
class scRNADataset(Dataset):
    """
    Manipulations of training data.
    
    Methods:
    According to Dataset class in PyTorch, the following methods must be implemented
        1. __init__
        2. __get__item
        3. __len__ 
    
   Adj_Generate: makes a sparse adjacency matrix given a TF_set.

    """
    def __init__(self, train_set, num_gene, flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
       #       TF, Target, Label
       #  ([[  46,  433,    1],
       #    [  46,  536,    1],
       #    [  46, 1117,    1]...]

        self.num_gene = num_gene
        # default False - the identifier whether to conduct causal inference
        self.flag = flag

    def __getitem__(self, idx):
        # input: idx = index

        # loads and returns a sample from the dataset at the given index idx
        # Example use case:
        # >>> train_load.__getitem__(2)
        #         (data, label)
        # (array([  46, 1117]), 1.0)

        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        if self.flag:
            # 65895
            train_len = len(train_label)
            train_tan = np.zeros([train_len,2])

            train_tan[:,0] = 1 - train_label
            train_tan[:,1] = train_label
            train_label = train_tan
           #      not label, label
           #      [[0., 1.],
           #       [0., 1.],
           #       [0., 1.]...

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)

    def Adj_Generate(self, TF_set, direction=False, loop=False):
        """
        Makes a sparse adjacency matrix given a TF_set.

        Args:
            direction: directed if True, o.w. undirected.
            loop: add self-loop in adjacency matrix if True.
        """
        # make a sparse matrix
        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)

        for pos in self.train_set:

            tf = pos[0]
            target = pos[1]

            # undirected edges -> make a symmetric matrix
            if direction == False:
                # there is an edge b/w tf and target
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    # make only 1 edge
                    adj[tf, target] = 1.0
                    # self-loop- target is the tf itself!
                    if target in TF_set:
                        adj[target, tf] = 1.0

        # adds 1 to the diagonal ~ each gene has a self-loop
        if loop:
            adj = adj + sp.identity(self.num_gene)

        #?# this line doesn't seem necessary.
        adj = adj.todok()
        return adj

#?# or maybe data is scaled, and not normalized? #?#
class load_data():
    # genes x cell type/condition. Entries: normalized single-cell values.
    # input example:
    #         RamDA_mESC_00h_A04  RamDA_mESC_00h_A05
    # HOPX    0   0
    # NFXL1   0.9311803422715608  0.7621028611988342
    # HDAC3   1.1279033808415402  1.339866172
    # CDC5L   2.0687816447980403  1.630139785

    def __init__(self, data, normalize=True):
        # pandas data frame
        self.data = data
        self.normalize = normalize

    def data_normalize(self, data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T

    def exp_data(self):
        # input: data
        # output: normalized data with type np.float32
        
        # Get all values from pandas dataframe (?) e.g. 
        # array([[0, 'HOPX', 0],
        #        [1, 'NFXL1', 1],
        #        [2, 'HDAC3', 2],
        data_feature = self.data.values

        #?# Why are they normalizing already normalized data? #?#
        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature

#?# Can better understand for next time!
def adj2sparse_tensor(adj):
    """
    Munging to convert adj (scipy.sparse._dok.dok_matrix) into 'torch.Tensor'.

    """
    # convert scipy.sparse._dok.dok_matrix -> scipy.sparse._coo.coo_matrix
    coo = adj.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor

def Evaluation(y_true, y_pred,flag=False):
    if flag:
        # y_p = torch.argmax(y_pred,dim=1)
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()


    y_t = y_true.cpu().numpy().flatten().astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)


    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    AUPR_norm = AUPR/np.mean(y_t)


    return AUC, AUPR, AUPR_norm



## Understood- self-explanatory ##
def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr


## Understood ##
def Network_Statistic(data_type,net_scale,net_type):
    # these are very sparse!
    if net_type =='STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale



    elif net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
               'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError































