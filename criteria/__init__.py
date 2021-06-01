### Standard DML criteria
from criteria import triplet, margin, proxynca
from criteria import contrastive, angular, arcface
from criteria import multisimilarity, quadruplet, oproxy
### Non-Standard Criteria
from criteria import moco, adversarial_separation, fast_moco, imrot, s2sd
### Basic Libs
import copy


"""================================================================================================="""
def select(loss, opt, to_optim=None, batchminer=None):
    losses = {'triplet': triplet,
              'margin':margin,
              'proxynca':proxynca,
              's2sd':s2sd,
              'angular':angular,
              'contrastive':contrastive,
              'oproxy':oproxy,
              'multisimilarity':multisimilarity,
              'arcface':arcface,
              'quadruplet':quadruplet,
              'adversarial_separation':adversarial_separation,
              'moco': moco,
              'imrot':imrot,
              'fast_moco':fast_moco}


    if loss not in losses: raise NotImplementedError('Loss {} not implemented!'.format(loss))

    loss_lib = losses[loss]
    if loss_lib.REQUIRES_BATCHMINER:
        if batchminer is None:
            raise Exception('Loss {} requires one of the following batch mining methods: {}'.format(loss, loss_lib.ALLOWED_MINING_OPS))
        else:
            if batchminer.name not in loss_lib.ALLOWED_MINING_OPS:
                raise Exception('{}-mining not allowed for {}-loss!'.format(batchminer.name, loss))

    loss_par_dict  = {'opt':opt}
    if loss_lib.REQUIRES_BATCHMINER:
        loss_par_dict['batchminer'] = batchminer

    criterion = loss_lib.Criterion(**loss_par_dict)

    if to_optim is not None:
        if loss_lib.REQUIRES_OPTIM:
            if hasattr(criterion,'optim_dict_list') and criterion.optim_dict_list is not None:
                to_optim += criterion.optim_dict_list
            else:
                to_optim    += [{'params':criterion.parameters(), 'lr':criterion.lr}]

        return criterion, to_optim
    else:
        return criterion
