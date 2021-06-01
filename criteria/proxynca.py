import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            opt: Namespace containing all relevant parameters.
        """
        super(Criterion, self).__init__()

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

        ####        
        self.num_proxies        = opt.n_classes
        self.embed_dim          = opt.embed_dim

        self.proxies            = torch.nn.Parameter(torch.randn(self.num_proxies, self.embed_dim)/8)
        self.class_idxs         = torch.arange(self.num_proxies)

        self.name           = 'proxynca'

        self.lr   = opt.lr * opt.loss_proxynca_lrmulti

        self.sphereradius = opt.loss_proxynca_sphereradius
        self.T            = opt.loss_proxynca_temperature
        self.convert_to_p = opt.loss_proxynca_convert_to_p
        self.cosine       = opt.loss_proxynca_cosine_dist
        self.sq_dist      = opt.loss_proxynca_sq_dist


    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        #Empirically, multiplying the embeddings during the computation of the loss seem to allow for more stable training;
        #presumably due to increased loss value.
        batch   = self.sphereradius*torch.nn.functional.normalize(batch, dim=1)
        proxies = self.sphereradius*torch.nn.functional.normalize(self.proxies, dim=1)

        #Loss based on distance to positive proxies
        if self.cosine:
            dist_to_pos_proxies = batch.unsqueeze(1).bmm(proxies[labels].unsqueeze(2)).squeeze(-1).squeeze(-1)
        else:
            if self.sq_dist:
                dist_to_pos_proxies = -(batch-proxies[labels]).pow(2).sum(-1).sqrt()
            else:
                dist_to_pos_proxies = -(batch-proxies[labels]).pow(2).sum(-1)

        loss_pos = torch.mean(-dist_to_pos_proxies/self.T)

        #Loss based on distance to negative (or all) proxies
        if not self.convert_to_p:
            batch_neg_idxs = labels.unsqueeze(1) != self.class_idxs.unsqueeze(1).T
        else:
            batch_neg_idxs = torch.ones((len(batch),self.num_proxies)).bool().to(labels.device)

        loss_neg = 0
        for neg_idxs, sample in zip(batch_neg_idxs, batch):
            if self.cosine:
                dist_to_neg_proxies = -sample.unsqueeze(0).mm(proxies[neg_idxs,:].T).squeeze(0)
            else:
                if self.sq_dist:
                    dist_to_neg_proxies = (sample.unsqueeze(0)-proxies[neg_idxs,:]).pow(2).sum(1).sqrt()
                else:
                    dist_to_neg_proxies = (sample.unsqueeze(0)-proxies[neg_idxs,:]).pow(2).sum(1)


            loss_neg           += torch.logsumexp(-dist_to_neg_proxies, dim=-1)
        loss_neg /= len(batch)

        loss = loss_pos + loss_neg
        # neg_proxies = torch.stack([torch.cat([self.class_idxs[:class_label],self.class_idxs[class_label+1:]]) for class_label in labels])
        # neg_proxies = torch.stack([proxies[neg_labels,:] for neg_labels in neg_proxies])
        # else:
        #     neg_proxies = torch.stack(.)
        #     #Compute Proxy-distances
        #     dist_to_neg_proxies = torch.sum((batch[:,None,:]-neg_proxies).pow(2),dim=-1)
        #     #Compute final proxy-based NCA loss
        #     negative_log_proxy_nca_loss = torch.mean(dist_to_pos_proxies[:,0]/self.T + torch.logsumexp(-dist_to_neg_proxies/self.T, dim=1))
        # else:
        #     norm_proxies =
        return loss
