import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


class BatchMiner():
    def __init__(self, opt):
        self.par          = opt
        self.lower_cutoff = opt.miner_distance_lower_cutoff
        self.upper_cutoff = opt.miner_distance_upper_cutoff
        self.name         = 'distance'

    def __call__(self, batch, labels, tar_labels=None, return_distances=False, distances=None):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs, dim = batch.shape

        if distances is None:
            distances = self.pdist(batch.detach()).clamp(min=self.lower_cutoff)
        sel_d = distances.shape[-1]

        positives, negatives = [],[]
        labels_visited       = []
        anchors              = []

        tar_labels = labels if tar_labels is None else tar_labels
        #
        # neg_all = labels.reshape(-1, 1) != tar_labels.reshape(1, -1)
        # pos_all = labels.reshape(-1, 1) == tar_labels.reshape(1, -1)
        #
        # log_q_d_inv = ((2.0 - float(dim)) * torch.log(distances) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (distances.pow(2))))
        # log_q_d_inv[pos_all] = 0
        # cop_q_d_inv = torch.exp(log_q_d_inv - log_q_d_inv.max(dim=1).values.reshape(-1, 1)) # - max(log) for stability
        # cop_q_d_inv[pos_all] = 0
        # cop_q_d_inv = cop_q_d_inv/cop_q_d_inv.sum(dim=1).reshape(-1, 1)
        # cop_q_d_inv = cop_q_d_inv.detach().cpu().numpy()

        for i in range(bs):
            neg = tar_labels!=labels[i]; pos = tar_labels==labels[i]

            anchors.append(i)
            q_d_inv = self.inverse_sphere_distances(dim, bs, distances[i], tar_labels, labels[i])
            negatives.append(np.random.choice(sel_d,p=q_d_inv))

            if np.sum(pos)>0:
                #Sample positives randomly
                if np.sum(pos)>1: pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))
                #Sample negatives by distance

        sampled_triplets = [[a,p,n] for a,p,n in zip(anchors, positives, negatives)]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets

    # def __call__(self, batch, labels, tar_labels=None, return_distances=False, distances=None):
    #     # if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
    #     bs, dim = batch.shape
    #
    #     import time
    #     start = time.time()
    #     if distances is None:
    #         distances = self.pdist(batch.detach()).clamp(min=self.lower_cutoff)
    #     sel_d = distances.shape[-1]
    #     print('A', time.time() - start)
    #     start = time.time()
    #     positives, negatives = [],[]
    #     labels_visited       = []
    #     anchors              = []
    #
    #     tar_labels = labels if tar_labels is None else tar_labels
    #
    #     pos_all = labels.view(-1, 1) == tar_labels.view(1, -1)
    #     pos_all_mul = ~pos_all
    #
    #     log_q_d_inv = ((2.0 - float(dim)) * torch.log(distances) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (distances.pow(2))))
    #
    #     print('A1', time.time() - start)
    #     start = time.time()
    #     log_q_d_inv = pos_all_mul * log_q_d_inv
    #     print('A2', time.time() - start)
    #     start = time.time()
    #     q_d_inv = torch.exp(log_q_d_inv - log_q_d_inv.max(dim=1).values.reshape(-1, 1)) # - max(log) for stability
    #     print('A3', time.time() - start)
    #     start = time.time()
    #     q_d_inv[pos_all] = 0
    #     print('A4', time.time() - start)
    #     start = time.time()
    #     q_d_inv = q_d_inv/q_d_inv.sum(dim=1).reshape(-1, 1)
    #     q_d_inv = q_d_inv.detach().cpu().numpy()
    #     print('A5', time.time() - start)
    #     start = time.time()
    #
    #     pos_all = pos_all.detach().cpu().numpy()
    #     for i in range(bs):
    #         pos = pos_all[i]
    #         anchors.append(i)
    #         negatives.append(np.random.choice(sel_d, p=q_d_inv[i]))
    #         if np.sum(pos)>0:
    #             #Sample positives randomly
    #             if np.sum(pos)>1: pos[i] = 0
    #             positives.append(np.random.choice(np.where(pos)[0]))
    #             #Sample negatives by distance
    #
    #     sampled_triplets = [[a,p,n] for a,p,n in zip(anchors, positives, negatives)]
    #
    #     print('B', time.time() - start)
    #
    #     if return_distances:
    #         return sampled_triplets, distances
    #     else:
    #         return sampled_triplets


    def inverse_sphere_distances(self, dim, bs, anchor_to_all_dists, labels, anchor_label):
            dists  = anchor_to_all_dists

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dists) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dists.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse. It can also lead to
            # errors where there are no available negatives (for high samples_per_class cases).
            # q_d_inv[np.where(dists.detach().cpu().numpy()>self.upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()


    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.sqrt()
