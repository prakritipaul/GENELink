"""
    Annotated with my comments
    2/6/24
"""

import sys
sys.path.insert(0, "/Users/prakritipaul/Git/GENELink/Code")

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from scGNN import GENELink
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2sparse_tensor, Evaluation,  Network_Statistic
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from PytorchTools import EarlyStopping
import numpy as np
import random
import glob
import os

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 90, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

"""
Non-Specific
mHSC-L learning rate = 3e-5
"""

# Data from:
# /Users/prakritipaul/Git/GENELink/Dataset/Benchmark Dataset/Specific Dataset/mESC/TFs+500

data_type = 'mESC'
num = 500
# Added this myself
net_type = "Specific"

def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    # Half sure about what index does.
    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)

density = Network_Statistic(data_type,num,net_type)

#### PART 1: GET FEATURES AND LABELS ####

data_dir = "/Users/prakritipaul/Git/GENELink/Demo/mESC/TFs+500/"
exp_file = data_dir + "BL--ExpressionData.csv"
tf_file = data_dir + "TF.csv"
target_file = data_dir + "Target.csv"

data_input = pd.read_csv(exp_file, index_col=0)
# pandas.core.frame.DataFrame with shape = 1120 x 421
#          RamDA_mESC_00h_A04  RamDA_mESC_00h_A05  ...  RamDA_mESC_72h_H11  RamDA_mESC_72h_H12
# BMI1               0.746461            0.932864  ...            1.539764            0.543634
# ARNT2              1.065002            0.401996  ...            0.000000            0.000000
# BTG2               0.328717            0.506675  ...            1.442061            1.306968
# MED4               1.559793            1.164828  ...            2.526651            1.739038
# SOAT1              0.743298            0.919495  ...            3.627453            4.072934
# ...                     ...                 ...  ...                 ...                 ...
# CREB3L2            0.119458            0.346268  ...            1.804374            2.185299
# ZBTB7A             1.124735            0.733272  ...            0.365299            0.111583
# MLXIP              1.202327            0.462298  ...            1.044437            1.039757
# GABBR1             2.445553            1.921987  ...            0.536102            0.494867
# ATF1               2.657772            2.803836  ...            2.956393            2.838447

loader = load_data(data_input)

#?# Step: Normalizes data #?#
feature = loader.exp_data()
# torch.Tensor with shape = 1120 x 421
# tensor([[-0.4145, -0.0988, -1.5430,  ..., -0.4367,  0.9292, -0.7580],
#         [ 2.0738,  0.3329,  2.1264,  ..., -0.7226, -0.7226, -0.7226],
#         [-0.5666, -0.3345, -0.6225,  ...,  0.7266,  0.8854,  0.7092],
#         ...,
#         [ 1.2108, -0.8548, -1.0150,  ...,  0.0263,  0.7701,  0.7570],
#         [ 1.7704,  1.1244,  2.0805,  ...,  0.7899, -0.5856, -0.6365],
#         [-0.6185, -0.2789,  0.9483,  ..., -0.5789,  0.0758, -0.1984]])

tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
# pd.read_csv(tf_file,index_col=0)
#           TF  index
# 0       BMI1      0
# 1      ARNT2      1
# 2       BTG2      2
# 3       MED4      3
# 4       TLE1      5
# ..       ...    ...
# 622  CREB3L2   1115
# 623   ZBTB7A   1116
# 624    MLXIP   1117
# 625   GABBR1   1118
# 626     ATF1   1119
#
# ->
# numpy.ndarray
# array([   0,    1,    2,    3,    5,    6,    8,    9,   13,   14,   17,
#           23,   24,   25,   26,   27,   29,   30,   31,  34,   35,   36...])

target = pd.read_csv(target_file, index_col=0)['index'].values.astype(np.int64)
#          Gene  index
# 0        BMI1      0
# 1       ARNT2      1
# 2        BTG2      2
# 3        MED4      3
# 4       SOAT1      4
# ...       ...    ...
# 1115  CREB3L2   1115
# 1116   ZBTB7A   1116
# 1117    MLXIP   1117
# 1118   GABBR1   1118
# 1119     ATF1   1119
#
# ->
# numpy.ndarray (as above)

## Step: Turn into torch.Tensors
feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)
# tensor([   0,    1,    2,    3,    5,    6,    8,    9,   13,   14,   17,   23,
#           24,   25,   26,   27,   29,   30,   31,   34,   35,   36,   38,   40...])

## Step: Use GPU if available. O.w. use CPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
tf = tf.to(device)

#### PART 2: GET TRAINING, TESTING, AND VALIDATION DATA (ndarrays) ####

train_test_dir = "/Users/prakritipaul/Git/GENELink/Demo/Train_validation_test/mESC 500/"
train_file = train_test_dir + "Train_set.csv"
test_file = train_test_dir + "Test_set.csv"
val_file = train_test_dir + "Validation_set.csv"

# Step: Make files where the embeddings will go.
# Note: There are 1120- 1 embedding/gene.
tf_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel1.csv'
target_embed_path = r'Result/'+data_type+' '+str(num)+'/Channel2.csv'

if not os.path.exists('Result/'+data_type+' '+str(num)):
    os.makedirs('Result/'+data_type+' '+str(num))

## Step: Get training, testing, and validation data
train_data = pd.read_csv(train_file, index_col=0).values
# numpy.ndarray
#     TF  Target  Label
# 0   46  433 1
# 1   46  536 1
# 2   46  1117    1
# 
# ->
#
# array([[  46,  433,    1],
       # [  46,  536,    1],
       # [  46, 1117,    1],

validation_data = pd.read_csv(val_file, index_col=0).values
# Same structure as above.

test_data = pd.read_csv(test_file, index_col=0).values
# Same structure as above.

#### PART 3: MAKE ADJACENCY MATRIX FROM TRAINING DATA ####

train_load = scRNADataset(train_data, feature.shape[0])
# 1120 x 1120
adj = train_load.Adj_Generate(tf, loop=args.loop)
adj = adj2sparse_tensor(adj)

#### PART 4: GET TRAINING, TESTING, AND VALIDATION DATA (tensors) ####
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)

#### PART 5: USE THE MODEL ####

model = GENELink(input_dim=feature.size()[1],
                hidden1_dim=args.hidden_dim[0],
                hidden2_dim=args.hidden_dim[1],
                hidden3_dim=args.hidden_dim[2],
                output_dim=args.output_dim,
                num_head1=args.num_head[0],
                num_head2=args.num_head[1],
                alpha=args.alpha,
                device=device,
                type=args.Type,
                reduction=args.reduction
                )


# adj = adj.to(device)
# model = model.to(device)
# train_data = train_data.to(device)
# test_data = test_data.to(device)
# validation_data = val_data.to(device)

# optimizer = Adam(model.parameters(), lr=args.lr)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

# model_path = 'model/'
# if not os.path.exists(model_path):
#     os.makedirs(model_path)


# for epoch in range(args.epochs):
#     running_loss = 0.0

#     for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
#         model.train()
#         optimizer.zero_grad()

#         if args.flag:
#             train_y = train_y.to(device)
#         else:
#             train_y = train_y.to(device).view(-1, 1)


#         # train_y = train_y.to(device).view(-1, 1)
#         pred = model(data_feature, adj, train_x)

#         #pred = torch.sigmoid(pred)
#         if args.flag:
#             pred = torch.softmax(pred, dim=1)
#         else:
#             pred = torch.sigmoid(pred)
#         loss_BCE = F.binary_cross_entropy(pred, train_y)


#         loss_BCE.backward()
#         optimizer.step()
#         scheduler.step()

#         running_loss += loss_BCE.item()


#     model.eval()
#     score = model(data_feature, adj, validation_data)
#     if args.flag:
#         score = torch.softmax(score, dim=1)
#     else:
#         score = torch.sigmoid(score)

#     # score = torch.sigmoid(score)

#     AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
#         #
#     print('Epoch:{}'.format(epoch + 1),
#             'train loss:{}'.format(running_loss),
#             'AUC:{:.3F}'.format(AUC),
#             'AUPR:{:.3F}'.format(AUPR))

# torch.save(model.state_dict(), model_path + data_type+' '+str(num)+'.pkl')

# model.load_state_dict(torch.load(model_path + data_type+' '+str(num)+'.pkl'))
# model.eval()
# tf_embed, target_embed = model.get_embedding()
# embed2file(tf_embed,target_embed,target_file,tf_embed_path,target_embed_path)

# score = model(data_feature, adj, test_data)
# if args.flag:
#     score = torch.softmax(score, dim=1)
# else:
#     score = torch.sigmoid(score)
# # score = torch.sigmoid(score)


# AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=args.flag)

# print('AUC:{}'.format(AUC),
#      'AUPRC:{}'.format(AUPR))






















