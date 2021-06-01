"""==================================================================================================="""
### Basic Libraries
import warnings
warnings.filterwarnings("ignore")
import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

### Repository-specific Libraries
import parameters    as par
import utilities.misc as misc


"""==================================================================================================="""
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
parser = par.s2sd_parameters(parser)
parser = par.maxentropy_parameters(parser)
parser.add_argument('--data_hardness', default=1, type=int, help='OOD-ness of datasets. Ranges/Default id are [CUB200-2011, (1-9), 3], [CARS196, (1-8), 1], [SOP, (1-8), 1]')
parser.add_argument('--new_test_dataset', default='None', type=str, help='Optionally evaluate on the test dataset of a different dataset.')
##### Read in parameters
opt = parser.parse_args()


"""==================================================================================================="""
### The following setting is useful when logging to wandb and running multiple seeds per setup:
### By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
if opt.savename=='group_plus_seed':
    if opt.log_online:
        opt.savename = opt.group+'_s{}'.format(opt.seed)
    else:
        opt.savename = ''


### Easy checkpointing
start_from_checkpoint = False
if opt.checkpoint:
    init_save_path = copy.deepcopy(opt.save_path)
    checkpath      = opt.save_path+'/'+opt.dataset+'/'+opt.savename
    if os.path.exists(checkpath):
        if any(['checkpoint.pth.tar'==x for x in os.listdir(checkpath)]):
            load_opt = pkl.load(open(checkpath+'/hypa.pkl', 'rb'))
            new_run  = False
            for key,item in vars(opt).items():
                if key!='load_to_ram':
                    if item!=vars(load_opt)[key] and key not in ['source_path', 'save_path', 'completed']:
                        new_run = True

            if not new_run:
                opt             = load_opt
                opt.save_path   = init_save_path
                opt.source_path = '/'.join(opt.source_path.split('/')[:-1])
                start_from_checkpoint = True

if opt.completed:
    print('\n\nTraining Run already completed!')
    exit()

### If wandb-logging is turned on, initialize the wandb-run here:
if opt.log_online:
    import wandb
    _ = os.system('wandb login {}'.format(opt.wandb_key))
    os.environ['WANDB_API_KEY'] = opt.wandb_key
    if not start_from_checkpoint:
        opt.unique_run_id = wandb.util.generate_id()
    wandb.init(id=opt.unique_run_id, resume='allow', project=opt.project, group=opt.group, name=opt.savename, dir=opt.source_path)
    wandb.config.update(opt)


"""==================================================================================================="""
### Load Remaining Libraries that need to be loaded after wandb
import torch, torch.nn as nn, torch.nn.functional as F
import torch.multiprocessing
import torchvision
torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler   as dsamplers
import datasets      as dataset_library
import criteria      as criteria
import metrics       as metrics
import batchminer    as bmine
import evaluation    as eval
from utilities import misc
from utilities import logger



"""==================================================================================================="""
full_training_start_time = time.time()

opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset

#Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

opt.pretrained      = not opt.not_pretrained
opt.evaluate_on_gpu = not opt.evaluate_on_cpu

os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu[0])

if start_from_checkpoint:
    chkp_data = torch.load(opt.save_path+'/'+opt.savename+'/checkpoint.pth.tar')
misc.set_seed(opt.seed)


"""==================================================================================================="""
##################### NETWORK SETUP ##################
opt.device = torch.device('cuda')
model      = archs.select(opt.arch, opt)

if opt.fc_lr<0:
    to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
else:
    all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'last_linear' not in x[0], model.named_parameters()))]
    fc_params         = model.model.last_linear.parameters()
    to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
                         {'params':fc_params,'lr':opt.fc_lr,'weight_decay':opt.decay}]

if start_from_checkpoint:
    opt       = chkp_data['opt']
    _ = model.load_state_dict(chkp_data['state_dict'])

_  = model.to(opt.device)


"""============================================================================"""
# Code needed for Uniform-Prior regularization (Sinha et al., 2020)
class Discriminator(nn.Module):
    def __init__(self, in_dim, latent):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, latent)
        self.fc2 = nn.Linear(latent, latent)
        self.fc3 = nn.Linear(latent, 1)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

if opt.with_entropy:
    discr = Discriminator(opt.maxentropy_chunksize, opt.maxentropy_latent)
    if start_from_checkpoint:
        _ = discr.load_state_dict(chkp_data['aux']['discr_state_dict'])
    _ = discr.to(opt.device)

    optimizer_discr     = torch.optim.Adam(discr.parameters(), lr=opt.maxentropy_lrmulti*opt.lr, betas=(0.5, 0.999))
    if start_from_checkpoint:
        optimizer_discr.load_state_dict(chkp_data['aux']['optimizer_discr_state_dict'])
    scheduler_discr = torch.optim.lr_scheduler.MultiStepLR(optimizer_discr, milestones=opt.maxentropy_tau, gamma=opt.maxentropy_gamma)
    if start_from_checkpoint:
        scheduler_discr.load_state_dict(chkp_data['aux']['scheduler_discr_state_dict'])




"""============================================================================"""
#################### DATALOADER SETUPS ##################
datasets    = dataset_library.select(opt.dataset, opt, opt.source_path)
if hasattr(model, 'specific_normalization'):
    for dataset_type in datasets.keys():
        if datasets[dataset_type] is not None:
            datasets[dataset_type].normal_transform_list[-1] = model.specific_normalization
            datasets[dataset_type].normal_transform = torchvision.transforms.Compose(
                datasets[dataset_type].normal_transform_list
            )

if opt.new_test_dataset != 'None':
    init_dataset = copy.deepcopy(opt.dataset)
    init_sourcepath = copy.deepcopy(opt.source_path)
    opt.source_path = opt.source_path.replace('/' + opt.dataset, '/' + opt.new_test_dataset)
    opt.dataset = opt.new_test_dataset
    datasets_temp = dataset_library.select(opt.new_test_dataset, opt, opt.source_path)
    datasets['testing'] = datasets_temp['testing']
    opt.dataset = init_dataset
    opt.source_path = init_sourcepath

dataloaders = {}
dataloaders['evaluation'] = torch.utils.data.DataLoader(datasets['evaluation'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
dataloaders['testing']    = torch.utils.data.DataLoader(datasets['testing'],    num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)

if opt.use_tv_split:
    dataloaders['validation'] = torch.utils.data.DataLoader(datasets['validation'], num_workers=opt.kernels, batch_size=opt.bs,shuffle=False)

train_data_sampler      = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict, datasets['training'].image_list)
if train_data_sampler.requires_storage:
    train_data_sampler.create_storage(dataloaders['evaluation'], model, opt.device)

dataloaders['training'] = torch.utils.data.DataLoader(datasets['training'], num_workers=opt.kernels, batch_sampler=train_data_sampler)

opt.n_classes  = len(dataloaders['training'].dataset.avail_classes)
opt.n_test_classes = len(dataloaders['testing'].dataset.avail_classes)




"""============================================================================"""
#################### CREATE LOGGING FILES ###############
sub_loggers = ['Train', 'Test', 'Model Grad']
if opt.use_tv_split: sub_loggers.append('Val')

if start_from_checkpoint:
    LOG = chkp_data['progress']
else:
    LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)


"""============================================================================"""
#################### Basic DML Loss Setup ####################
batchminer   = bmine.select(opt.batch_mining, opt)
criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)

if start_from_checkpoint:
    _ = criterion.load_state_dict(chkp_data['aux']['criterion'][-1])

_ = criterion.to(opt.device)

if 'criterion' in train_data_sampler.name:
    train_data_sampler.internal_criterion = criterion


"""============================================================================"""
#################### OPTIM SETUP ####################
if opt.optim == 'adam':
    optimizer    = torch.optim.Adam(to_optim)
else:
    raise Exception('Optimizer <{}> not available!'.format(opt.optim))

if start_from_checkpoint:
    _ = optimizer.load_state_dict(chkp_data['aux']['optimizer'])

scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
if start_from_checkpoint:
    scheduler.load_state_dict(chkp_data['aux']['scheduler'])


"""============================================================================"""
#################### METRIC COMPUTER ####################
metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)


"""============================================================================"""
################### Summary #########################3
chkp_text  = 'RESTARTED FROM CHKP: {} {}\n'.format(start_from_checkpoint, '- at epoch {}'.format(opt.epoch+1) if 'epoch' in vars(opt) else '')
data_text  = 'Dataset:\t {}'.format(opt.dataset.upper())
setup_text = 'Objective:\t {}'.format(opt.loss.upper())
miner_text = 'Batchminer:\t {}'.format(opt.batch_mining if criterion.REQUIRES_BATCHMINER else 'N/A')
arch_text  = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
summary    = chkp_text+'\n'+data_text+'\n'+setup_text+'\n'+miner_text+'\n'+arch_text
print(summary)


"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
include_aux_augmentations = dataloaders['training'].dataset.include_aux_augmentations
loss_args  = {'batch':None, 'labels':None, 'batch_features':None, 'f_embed':None}
init_lrs = [param_group['lr'] for param_group in optimizer.param_groups[1:]]

if not start_from_checkpoint:
    opt.epoch  = 0
else:
    if 'epoch' in vars(opt):
        opt.epoch += 1
epochs = range(opt.epoch, opt.n_epochs)

for epoch in epochs:
    opt.epoch = epoch
    epoch_start_time = time.time()
    #Set seeds for each epoch - this ensures reproducibility after resumption.
    misc.set_seed(opt.n_epochs*opt.seed+epoch)

    ### Scheduling Changes specifically for cosine scheduling
    if opt.scheduler!='none':
        print('Running with learning rates {}...'.format(' | '.join('{}'.format(x['lr']) for x in optimizer.param_groups)))

    ### Train one epoch
    start = time.time()
    _ = model.train()

    loss_collect = []
    data_iterator = tqdm(dataloaders['training'], desc='Epoch {} Training...'.format(epoch))

    for i,out in enumerate(data_iterator):
        class_labels, input, input_indices = out

        input      = input.to(opt.device)
        model_args = {'x':input.to(opt.device), 'warmup':epoch<opt.warmup}

        out_dict  = model(**model_args)
        embeds, avg_features, features, extra_embeds = [out_dict[key] for key in ['embeds', 'avg_features', 'features', 'extra_embeds']]

        loss_args['input_batch']    = input
        loss_args['batch']          = embeds
        loss_args['extra_batches']  = extra_embeds
        loss_args['labels']         = class_labels
        loss_args['f_embed']        = model.model.last_linear
        loss_args['batch_features'] = features
        loss_args['avg_batch_features'] = avg_features
        loss_args['model']          = model
        loss_args['optim']          = optimizer

        loss      = criterion(**loss_args)

        if opt.with_entropy:
            # Apply adversarial uniformity regularization.
            if iter_count % opt.maxentropy_iter == 0:
                # DSC STEP
                embeddings = avg_features.view(-1, opt.maxentropy_chunksize).detach()
                z          = torch.rand(embeddings.size(0), opt.maxentropy_chunksize).to(opt.device) * 2. - 1.
                ones       = torch.ones(embeddings.size(0), 1).to(opt.device)
                preds_real = discr(z)
                real_loss  = F.binary_cross_entropy(preds_real, ones)

                zeros      = torch.zeros(embeddings.size(0), 1).to(opt.device)
                preds_fake = discr(embeddings.detach())
                fake_loss  = F.binary_cross_entropy(preds_fake, zeros)
                dsc_loss   = real_loss + fake_loss

                optimizer_discr.zero_grad()
                dsc_loss.backward()
                optimizer_discr.step()

            embeddings = avg_features.view(-1, opt.maxentropy_chunksize)
            ones       = torch.ones(embeddings.size(0), 1).to(opt.device)
            preds_fake = discr(embeddings)
            adv_loss   = F.binary_cross_entropy(preds_fake, ones)
            loss += opt.maxentropy_w * adv_loss

        """========================="""
        optimizer.zero_grad()
        loss.backward()

        ### Compute Model Gradients and log them!
        grads = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))

        LOG.progress_saver['Model Grad'].log('Grad L2',  grad_l2,  group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

        ### Update network weights!
        optimizer.step()

        ###
        loss_collect.append(loss.item())

        ###
        iter_count += 1

        data_iterator.set_postfix_str('Loss: {0:.4f}'.format(np.mean(loss_collect)))
        if i==len(dataloaders['training'])-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))


    ####
    result_metrics = {'loss': np.mean(loss_collect)}

    ####
    LOG.progress_saver['Train'].log('epochs', epoch)
    for metricname, metricval in result_metrics.items():
        LOG.progress_saver['Train'].log(metricname, metricval)
    LOG.progress_saver['Train'].log('time', np.round(time.time()-start, 4))

    ### Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()
    if opt.with_entropy:
        scheduler_discr.step()

    """======================================="""
    ### Evaluate Metric for Training & Test (& Validation)
    _ = model.eval()
    print('\nComputing Testing Metrics...')

    aux_store = {'optimizer':optimizer.state_dict(), 'criterion':(criterion, criterion.state_dict()),
                 'scheduler':scheduler.state_dict(), 'dataloaders':dataloaders,
                 'discr_state_dict':discr.state_dict() if opt.checkpoint and opt.with_entropy else None,
                 'optimizer_discr_state_dict':optimizer_discr.state_dict() if opt.checkpoint and opt.with_entropy else None,
                 'scheduler_discr_state_dict':scheduler_discr.state_dict() if opt.checkpoint and opt.with_entropy else None,
                 'datasets':datasets, 'train_data_sampler':train_data_sampler}

    eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['testing']], model, opt, opt.evaltypes, opt.device, log_key='Test', aux_store=aux_store)
    if opt.use_tv_split:
        print('\nComputing Validation Metrics...')
        eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['validation']], model, opt, opt.evaltypes, opt.device, log_key='Val', aux_store=aux_store)

    if not opt.no_train_metrics:
        print('\nComputing Training Metrics...')
        eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['evaluation']], model, opt, opt.evaltypes, opt.device, log_key='Train', aux_store=aux_store)

    import wandb
    LOG.update(all=True)

    eval.set_checkpoint(model, opt, LOG, LOG.prop.save_path+'/checkpoint.pth.tar', aux=aux_store)

    print('\nTotal Epoch Runtime: {0:4.2f}s'.format(time.time()-epoch_start_time))
    print('\n-----\n')


"""======================================================="""
opt.completed = True
pkl.dump(opt,open(opt.save_path+"/hypa.pkl","wb"))
