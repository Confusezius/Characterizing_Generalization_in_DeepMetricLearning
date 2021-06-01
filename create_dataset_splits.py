import warnings
warnings.filterwarnings("ignore")
import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
sys.path.insert(0, '..')
import time, pickle as pkl, random, json, collections

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm

import architectures as archs
import datasets as datasets
import metrics as metrics
from utilities import misc
import parameters as par
import utilities.misc as misc
import split_helpers as helper


"""==================================================================================================="""
parser = argparse.ArgumentParser()
parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
parser.add_argument('--n_swaps',   default=25, type=int)
parser.add_argument('--swaps_iter', default=2,  type=int)
parser.add_argument('--load',      action='store_true')
parser.add_argument('--super', action='store_true')
##### Read in parameters
# Run with e.g. python create_dataset_splits.py --dataset cub200 [cars196, onlihe_products].
opt = parser.parse_args()
# Note: For SOP, set e.g. opt.swaps_iter = 1000 and opt.n_swaps=20, respectively.


"""==================================================================================================="""
def set_seed(seed):
    torch.backends.cudnn.deterministic=True;
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(opt.seed)

os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu[0])

opt.device = torch.device('cuda')
model      = archs.select(opt.arch, opt)
_          = model.to(opt.device)

dataloaders = {}
if opt.dataset=='online_products':
    opt.source_path += '/'+opt.dataset
    datasets    = datasets.select(opt.dataset, opt, opt.source_path)
else:
    datasets    = datasets.select(opt.dataset, opt, opt.source_path+'/'+opt.dataset)
dataloaders['training'] = torch.utils.data.DataLoader(datasets['evaluation'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
dataloaders['testing']  = torch.utils.data.DataLoader(datasets['testing'],    num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)


"""==================================================================================================="""
info_dict = {}
feat_collect         = []
class_labels_collect = []
img_paths_collect    = []

# These are the splits and FIDs used in our experiments.
# Note that due to internal differences between the initial script and this public one,
# there won't be an EXACT matching, but shifts/splits will be very close.
splits_to_use = {
    'cub200': {
        'id': ['-20', '-10', '0', '6', '10', '30', 'R22', 'R48', 'R66'],
        'base_id': ['-20', '-10', '0', '6', '10', '30'],
        'fid': [19.16, 28.49, 52.62, 72.20, 92.48, 120.38, 136.45, 152.04, 173.94]},
    'cars196': {
        'id': ['0', '6', '16', '20', '30', 'R18', 'R42', 'R66'],
        'base_id': ['0', '6', '16', '20', '30'],
        'fid': [8.59, 14.33, 32.18, 43.58, 63.29, 86.48, 101.17, 123.03]},
    'online_products': {
        'id': ['0', '1000', '2000', '3000', '4000', '5000', 'R2000', 'R6000'],
        'base_id': ['0', '1000', '2000', '3000', '4000', '5000'],
        'fid': [3.43, 24.59, 53.47, 99.38, 135.53, 155.25, 189.81, 235.10]}
}

if not opt.load:
    info_dict = helper.get_features(model, dataloaders, opt.dataset, opt.device)
    # Save dictionaries of features and classmeans.
    pkl.dump(info_dict,open('{}_dict.pkl'.format(opt.dataset),'wb'))
else:
    # If chosen, load pretrained embedding dictionaries.
    info_dict = pkl.load(open('{}_dict.pkl'.format(opt.dataset),'rb'))
    print("Data loaded!\n")


"""==============================================================="""
# If opt.super is set, swap classes by superclass context.
if opt.super:
    train_classmean_feats = info_dict['training']['classmeans_super']['feats']
    train_classmean_cls   = info_dict['training']['classmeans_super']['classes']
    test_classmean_feats  = info_dict['testing']['classmeans_super']['feats']
    test_classmean_cls    = info_dict['testing']['classmeans_super']['classes']
else:
    train_classmean_feats = info_dict['training']['classmeans']['feats']
    train_classmean_cls   = info_dict['training']['classmeans']['classes']
    test_classmean_feats  = info_dict['testing']['classmeans']['feats']
    test_classmean_cls    = info_dict['testing']['classmeans']['classes']


# Generate harder (more OOD) splits with class swapping.
hard_SPLITS, hard_fids, hard_final_feats = helper.split_maker(
    copy.deepcopy(train_classmean_feats), copy.deepcopy(train_classmean_cls),
    copy.deepcopy(test_classmean_feats), copy.deepcopy(test_classmean_cls),
    N_SWAPS=opt.n_swaps, SWAPS_PER_ITER=opt.swaps_iter, HISTORY=5, inverse=False
)

# Generate harder splits via class removal.
hard_removed_SPLITS, hard_removed_fids = helper.split_maker_with_class_removal(
    copy.deepcopy(hard_final_feats['train']), copy.deepcopy(hard_SPLITS[48]['train']),
    copy.deepcopy(hard_final_feats['test']), copy.deepcopy(hard_SPLITS[48]['test']),
    N_SWAPS=opt.n_swaps+10, SWAPS_PER_ITER=opt.swaps_iter, HISTORY=5, inverse=False
)

# Generate easier (less OOD) splits with class swapping.
if opt.dataset == 'cub200':
    easy_SPLITS, easy_fids, _ = helper.split_maker(
        copy.deepcopy(train_classmean_feats), copy.deepcopy(train_classmean_cls),
        copy.deepcopy(test_classmean_feats), copy.deepcopy(test_classmean_cls),
        N_SWAPS=opt.n_swaps-10, SWAPS_PER_ITER=opt.swaps_iter, HISTORY=30, inverse=True
    )

SPLITS = {}
for key in hard_SPLITS.keys():
    SPLITS[key] = {}
    SPLITS[key]['train'] = sorted(hard_SPLITS[key]['train'])
    SPLITS[key]['test']  = sorted(hard_SPLITS[key]['test'])
    SPLITS[key]['fid']   = hard_SPLITS[key]['fid']
if opt.dataset == 'cub200':
    for key in easy_SPLITS.keys():
        if key not in SPLITS.keys():
            SPLITS[key] = {}
            SPLITS[key]['train'] = sorted(easy_SPLITS[key]['train'])
            SPLITS[key]['test']  = sorted(easy_SPLITS[key]['test'])
            SPLITS[key]['fid']   = easy_SPLITS[key]['fid']


"""==============================================================="""
# Only select the splits that are going to be used for the experiments.
merged_dict = {}
for i, idx in enumerate(splits_to_use[opt.dataset]['id']):
    if 'R' not in idx:
        if '-' in idx:
            idx = int(idx)
            merged_dict[i+1] = easy_SPLIT[idx]
        else:
            idx = int(idx)
            merged_dict[i+1] = hard_SPLIT[idx]
    else:
        idx = idx.replace('R', '')
        idx = int(idx)
        merged_dict[i+1] = hard_removed_SPLITS[idx]

# Save complete split dictionary.
pkl.dump(merged_dict, open('{}{}_splits.pkl'.format('super_' if opt.super else '', opt.dataset),'wb'))
