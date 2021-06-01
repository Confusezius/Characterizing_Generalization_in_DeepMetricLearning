import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from scipy import linalg
from sklearn.metrics import pairwise_distances
import torch
import torch.nn as nn
from tqdm import tqdm
import umap

def get_features(model, dataloaders, dataset, device, return_all=False):
    info_dict = {}
    feat_collect         = []
    class_labels_collect = []
    img_paths_collect    = []

    _ = model.eval()
    for split in ['training','testing']:
        # info_dict[split] = {'classnames':[], 'paths':[], 'labels':[]}
        info_dict[split] = {'feats':[], 'classnames':[], 'paths':[], 'labels':[]}
        if dataset == 'online_products': info_dict[split]['classnames_super'] = []

        data_iterator        = tqdm(dataloaders[split], desc='Extracting features for [{}]...'.format(split))

        with torch.no_grad():
            for i, out in enumerate(data_iterator):
                class_labels, input, input_indices = out
                img_paths = [dataloaders[split].dataset.image_list[i][0] for i in input_indices.detach().cpu().numpy()]

                input      = input.to(device)
                model_args = {'x': input.to(device)}
                out_dict   = model(**model_args)
                embeds, avg_features, _, _ = [out_dict[key] for key in ['embeds', 'avg_features', 'features', 'extra_embeds']]

                info_dict[split]['feats'].extend(list(avg_features.detach().cpu().numpy()))
                info_dict[split]['paths'].extend(img_paths)
                if dataset == 'online_products':
                    info_dict[split]['classnames'].extend([x.split('/')[-1].split('_')[0] for x in img_paths])
                    info_dict[split]['classnames_super'].extend([x.split('/')[-2] for x in img_paths])
                else:
                    info_dict[split]['classnames'].extend([x.split('/')[-2] for x in img_paths])
                info_dict[split]['labels'].extend(list(class_labels.detach().cpu().numpy()))

        for key in info_dict[split].keys():
            info_dict[split][key] = np.stack(info_dict[split][key])

        info_dict[split]['classmeans'] = {'feats':[], 'classes':[]}

        for lab in np.unique(info_dict[split]['classnames']):
            lab_samples = np.where(info_dict[split]['classnames']==lab)[0]
            info_dict[split]['classmeans']['feats'].append(np.mean(info_dict[split]['feats'][lab_samples],axis=0))
            info_dict[split]['classmeans']['classes'].append(lab)

        if dataset == 'online_products':
            info_dict[split]['classmeans_super'] = {'feats':[], 'classes':[]}
            for lab in np.unique(info_dict[split]['classnames_super']):
                lab_samples = np.where(info_dict[split]['classnames_super']==lab)[0]
                info_dict[split]['classmeans_super']['feats'].append(np.mean(info_dict[split]['feats'][lab_samples],axis=0))
                info_dict[split]['classmeans_super']['classes'].append(lab)

        if not return_all:
            del info_dict[split]['feats']

    return info_dict

def metric_fid(train_feats, test_feats, eps=1e-8):

    total_feats  = np.concatenate([train_feats, test_feats],axis=0)
    total_labels = np.concatenate([np.ones(len(train_feats)), np.zeros(len(test_feats))], axis=0).astype('int32')

    def stats(features):
        return np.mean(features,axis=0), np.cov(features, rowvar=False)

    mu_tr, cov_tr = stats(train_feats)
    mu_ts, cov_ts = stats(test_feats)

    diff      = mu_tr - mu_ts
    diag_offs = np.eye(len(cov_ts))*eps
    covmean   = linalg.sqrtm((cov_tr+diag_offs).dot((cov_ts+diag_offs)), disp=False)[0].real
    fid       = diff.dot(diff) + np.trace(cov_tr) + np.trace(cov_ts) - 2 * np.trace(covmean)

    return fid

def plot_umap(feats, labs, savename):
    mapper = umap.UMAP()
    mapped_feats = mapper.fit_transform(feats)
    f,ax = plt.subplots(1)
    ax.plot(mapped_feats[:len(train_classmean_feats),0],mapped_feats[:len(train_classmean_feats),1],'b.',label='Train')
    ax.plot(mapped_feats[len(train_classmean_feats):,0],mapped_feats[len(train_classmean_feats):,1],'r.',label='Test')
    ax.legend()
    f.tight_layout()
    f.savefig(savename)
    plt.close()

def split_maker(train_feats, train_cls, test_feats, test_cls, N_SWAPS=51, SWAPS_PER_ITER=1, HISTORY=5, inverse=False):
    train_already_swapped, test_already_swapped = [], []
    SPLITS, FIDS  = {},[]

    iterator = range(N_SWAPS)

    for i in iterator:
        start = time.time()

        ix         = -i if inverse else i
        ix        *= SWAPS_PER_ITER
        SPLITS[ix] = {'train':copy.deepcopy(train_cls), 'test':copy.deepcopy(test_cls)}
        fid        = metric_fid(np.stack(train_feats), np.stack(test_feats))

        SPLITS[ix]['fid'] = fid
        final_feats = {'train': train_feats, 'test': test_feats}

        FIDS.append(fid)
        print('FID after {0} swaps: {1:4.4f}'.format(ix, fid))

        # for k in range(SWAPS_PER_ITER):
        trainmean = np.mean(train_feats,axis=0)
        testmean  = np.mean(test_feats,axis=0)

        # start = time.time()
        dists_train_trainmean = pairwise_distances(np.stack(train_feats), trainmean.reshape(1,-1), metric='euclidean')
        dists_train_testmean  = pairwise_distances(np.stack(train_feats), testmean.reshape(1,-1),  metric='euclidean')
        dists_test_trainmean  = pairwise_distances(np.stack(test_feats),  trainmean.reshape(1,-1), metric='euclidean')
        dists_test_testmean   = pairwise_distances(np.stack(test_feats),  testmean.reshape(1,-1),  metric='euclidean')
        # print(time.time()-start)

        train_swaps = np.argsort((dists_train_testmean - dists_train_trainmean).reshape(-1))
        test_swaps  = np.argsort((dists_test_trainmean - dists_test_testmean).reshape(-1))

        for k in range(SWAPS_PER_ITER):
            swapped_train, swapped_test = False, False
            if inverse:
                for train_swap_temp, test_swap_temp in zip(train_swaps[::-1][k:], test_swaps[::-1][k:]):
                    if train_swap_temp not in train_already_swapped[-HISTORY:] and not swapped_train:
                        train_already_swapped.append(train_swap_temp)
                        train_swap    = train_swap_temp
                        swapped_train = True
                    if test_swap_temp not in test_already_swapped[-HISTORY:] and not swapped_test:
                        test_already_swapped.append(test_swap_temp)
                        test_swap    = test_swap_temp
                        swapped_test = True
                    if swapped_train and swapped_test:
                        break
            else:
                for train_swap_temp, test_swap_temp in zip(train_swaps[k:], test_swaps[k:]):
                    if train_swap_temp not in train_already_swapped[-HISTORY:] and not swapped_train:
                        train_already_swapped.append(train_swap_temp)
                        train_swap    = train_swap_temp
                        swapped_train = True
                    if test_swap_temp not in test_already_swapped[-HISTORY:] and not swapped_test:
                        test_already_swapped.append(test_swap_temp)
                        test_swap = test_swap_temp
                        swapped_test = True
                    if swapped_train and swapped_test:
                        break

            train_feats[train_swap], test_feats[test_swap] = test_feats[test_swap], train_feats[train_swap]
            train_cls[train_swap],   test_cls[test_swap]   = test_cls[test_swap],   train_cls[train_swap]


    return SPLITS, FIDS, final_feats




def split_maker_with_class_removal(train_feats, train_cls, test_feats, test_cls, N_SWAPS=51, SWAPS_PER_ITER=1, HISTORY=5, inverse=False):
    train_already_swapped, test_already_swapped = [], []
    SPLITS, FIDS  = {},[]

    iterator = range(N_SWAPS)

    for i in iterator:
        start = time.time()

        ix         = -i if inverse else i
        ix        *= SWAPS_PER_ITER
        SPLITS[ix] = {'train':copy.deepcopy(train_cls), 'test':copy.deepcopy(test_cls)}
        fid        = metric_fid(np.stack(train_feats), np.stack(test_feats))

        SPLITS[ix]['fid'] = fid

        final_feats = {'train': train_feats, 'test': test_feats}

        FIDS.append(fid)
        print('FID after {0} swaps: {1:4.4f}'.format(ix, fid))

        # for k in range(SWAPS_PER_ITER):
        trainmean = np.mean(train_feats,axis=0)
        testmean  = np.mean(test_feats,axis=0)

        # start = time.time()
        dists_train_trainmean = pairwise_distances(np.stack(train_feats), trainmean.reshape(1,-1), metric='euclidean')
        dists_train_testmean  = pairwise_distances(np.stack(train_feats), testmean.reshape(1,-1),  metric='euclidean')
        dists_test_trainmean  = pairwise_distances(np.stack(test_feats),  trainmean.reshape(1,-1), metric='euclidean')
        dists_test_testmean   = pairwise_distances(np.stack(test_feats),  testmean.reshape(1,-1),  metric='euclidean')
        # print(time.time()-start)

        old_train_feats, old_test_feats = copy.deepcopy(train_feats), copy.deepcopy(test_feats)
        old_train_cls, old_test_cls = copy.deepcopy(train_cls), copy.deepcopy(test_cls)

        train_swaps = np.argsort((dists_train_testmean - dists_train_trainmean).reshape(-1))
        test_swaps  = np.argsort((dists_test_trainmean - dists_test_testmean).reshape(-1))

        train_swap_coll = []
        test_swap_coll = []
        for k in range(SWAPS_PER_ITER):
            swapped_train, swapped_test = False, False
            if inverse:
                for train_swap_temp, test_swap_temp in zip(train_swaps[::-1][k:], test_swaps[::-1][k:]):
                    if train_swap_temp not in train_already_swapped[-HISTORY:] and not swapped_train:
                        train_already_swapped.append(train_swap_temp)
                        train_swap    = train_swap_temp
                        swapped_train = True
                    if test_swap_temp not in test_already_swapped[-HISTORY:] and not swapped_test:
                        test_already_swapped.append(test_swap_temp)
                        test_swap    = test_swap_temp
                        swapped_test = True
                    if swapped_train and swapped_test:
                        break
            else:
                for train_swap_temp, test_swap_temp in zip(train_swaps[k:], test_swaps[k:]):
                    if train_swap_temp not in train_already_swapped[-HISTORY:] and not swapped_train:
                        train_already_swapped.append(train_swap_temp)
                        train_swap    = train_swap_temp
                        swapped_train = True
                    if test_swap_temp not in test_already_swapped[-HISTORY:] and not swapped_test:
                        test_already_swapped.append(test_swap_temp)
                        test_swap = test_swap_temp
                        swapped_test = True
                    if swapped_train and swapped_test:
                        break

            train_swap_coll.append(train_swap)
            test_swap_coll.append(test_swap)

            train_feats[train_swap], test_feats[test_swap] = test_feats[test_swap], train_feats[train_swap]
            train_cls[train_swap],   test_cls[test_swap]   = test_cls[test_swap],   train_cls[train_swap]

        # Move this correction to an additional function to add on top.
        adjusted_fid = metric_fid(np.stack(train_feats), np.stack(test_feats))

        if adjusted_fid >= fid:
            train_feats, test_feats = old_train_feats, old_test_feats
            train_cls, test_cls = old_train_cls, old_test_cls

            train_swap = np.array(sorted(list(set(train_swap_coll)), reverse=True))
            test_swap = np.array(sorted(list(set(test_swap_coll)), reverse=True))
            for ix in train_swap:
                del train_feats[ix]
                del train_cls[ix]
            for ix in test_swap:
                del test_feats[ix]
                del test_cls[ix]

    return SPLITS, FIDS
