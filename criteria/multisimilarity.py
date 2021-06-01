import numpy as np
import copy
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
import copy


"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, **kwargs):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(Criterion, self).__init__()
        self.pars = opt


        self.n_classes          = opt.n_classes

        self.pos_weight = opt.loss_multisimilarity_pos_weight
        self.neg_weight = opt.loss_multisimilarity_neg_weight
        self.margin     = opt.loss_multisimilarity_margin
        self.pos_thresh = opt.loss_multisimilarity_pos_thresh
        self.neg_thresh = opt.loss_multisimilarity_neg_thresh
        self.d_mode     = opt.loss_multisimilarity_d_mode
        self.name       = 'multisimilarity'

        self.lr = opt.lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        bs = len(batch)
        self.dim = 0
        self.embed_dim = batch.shape[-1]
        self.similarity = self.smat(batch, batch, self.d_mode)

        ###
        if self.d_mode=='euclidean':
            pos_weight = -1.*self.pos_weight
            neg_weight = -1.*self.neg_weight
        else:
            pos_weight = self.pos_weight
            neg_weight = self.neg_weight

        ###
        w_pos_sims = -pos_weight*(self.similarity-self.pos_thresh)
        w_neg_sims =  neg_weight*(self.similarity-self.neg_thresh)

        ###
        labels   = labels.unsqueeze(1)
        self.bsame_labels = (labels.T == labels.view(-1,1)).to(batch.device).T
        self.bdiff_labels = (labels.T != labels.view(-1,1)).to(batch.device).T

        ### Compute MultiSimLoss
        pos_mask, neg_mask = self.sample_mask(self.similarity)
        self.pos_mask, self.neg_mask = pos_mask, neg_mask

        pos_s = self.masked_logsumexp(w_pos_sims, mask=pos_mask, dim=self.dim, max=True  if self.d_mode=='euclidean' else False)
        neg_s = self.masked_logsumexp(w_neg_sims, mask=neg_mask, dim=self.dim, max=False if self.d_mode=='euclidean' else True)

        ###
        pos_s, neg_s = 1./np.abs(pos_weight)*torch.nn.Softplus()(pos_s), 1./np.abs(neg_weight)*torch.nn.Softplus()(neg_s)
        pos_s, neg_s = pos_s.mean(), neg_s.mean()
        loss = pos_s + neg_s


        return loss


    ###
    def sample_mask(self, sims):
        ### Get Indices/Sampling Bounds
        bsame_labels = copy.deepcopy(self.bsame_labels)
        bdiff_labels = copy.deepcopy(self.bdiff_labels)
        pos_bound, neg_bound = [], []
        bound = []
        for i in range(len(sims)):
            pos_ixs    = bsame_labels[i]
            neg_ixs    = bdiff_labels[i]
            pos_ixs[i] = False
            pos_bsims  = self.similarity[i][pos_ixs]
            neg_bsims  = self.similarity[i][neg_ixs]
            if self.d_mode=='euclidean':
                pos_bound.append(pos_bsims.max())
                neg_bound.append(neg_bsims.min())
            else:
                pos_bound.append(pos_bsims.min())
                neg_bound.append(neg_bsims.max())
        pos_bound, neg_bound = torch.stack(pos_bound), torch.stack(neg_bound)
        ### Get LogSumExp-Masks
        if self.d_mode=='euclidean':
            self.neg_mask = neg_mask = self.bdiff_labels*((self.similarity - self.margin) < pos_bound)
            self.pos_mask = pos_mask = self.bsame_labels*((self.similarity + self.margin) > neg_bound)
        else:
            self.neg_mask = neg_mask = self.bdiff_labels*((self.similarity + self.margin) > pos_bound)
            self.pos_mask = pos_mask = self.bsame_labels*((self.similarity - self.margin) < neg_bound)

        return pos_mask, neg_mask


    ###
    def smat(self, A, B, mode='cosine'):
        if mode=='cosine':
            return A.mm(B.T)
        elif mode=='euclidean':
            return (A.mm(A.T).diag().unsqueeze(-1)+B.mm(B.T).diag().unsqueeze(0)-2*A.mm(B.T)).clamp(min=1e-20).sqrt()


    ###
    def masked_logsumexp(self, sims, dim=0, mask=None, max=True):
        if mask is None:
            return torch.logsumexp(sims, dim=dim)
        else:
            if not max:
                ref_v      = (sims*mask).min(dim=dim, keepdim=True)[0]
            else:
                ref_v      = (sims*mask).max(dim=dim, keepdim=True)[0]

            nz_entries = (sims*mask)
            nz_entries = nz_entries.max(dim=dim,keepdim=True)[0]+nz_entries.min(dim=dim,keepdim=True)[0]
            nz_entries = torch.where(nz_entries.view(-1))[0].view(-1)

            if not len(nz_entries):
                return torch.tensor(0).to(torch.float).to(sims.device)
            else:
                return torch.log((torch.sum(torch.exp(sims-ref_v.detach())*mask,dim=dim)).view(-1)[nz_entries])+ref_v.detach().view(-1)[nz_entries]
