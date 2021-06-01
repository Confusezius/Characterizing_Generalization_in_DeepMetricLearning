"""==================================================================================================="""
### Basic Libraries
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
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
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()
parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
parser = par.diva_parameters(parser)
#
parser.add_argument('--data_hardness', default=1, type=int, help='OOD-ness of datasets. Ranges/Default id are [CUB200-2011, (1-9), 3], [CARS196, (1-8), 1], [SOP, (1-8), 1]')
parser.add_argument('--new_test_dataset', default='None', type=str, help='Optionally evaluate on the test dataset of a different dataset.')
# Few-Shot parameters ('finetune')
parser.add_argument('--finetune_lr_multi', default=3, type=float, help='Multiplier to the baseline learning rate used at test time for optional few-shot adaptation.')
parser.add_argument('--finetune_iter', default=20, type=int, help='Number of finetuning iterations.')
parser.add_argument('--finetune_shots', default=0, type=int, help='If non-zero, determines the number of samples/class given for few-shot adaptation at test time.')
parser.add_argument('--finetune_only_last', action='store_true', help='Flag. If set, finetuning is only performed for the very last linear layer of the embedding network.')
parser.add_argument('--recall_only', action='store_true', help='Flag. If set, finetuning is only performed for the very last linear layer of the embedding network.')
parser.add_argument('--finetune_optim', default='adam', type=str, help='If non-zero, determines the number of samples/class given for few-shot adaptation at test time.')
parser.add_argument('--finetune_criterion', default='margin', type=str, help='If non-zero, determines the number of samples/class given for few-shot adaptation at test time.')
parser.add_argument('--finetune_reweight', action='store_true', help='Flag. If set, finetuning is only performed for the very last linear layer of the embedding network.')
##### Read in parameters
opt = parser.parse_args()

# DiVA-specific parameters.
opt.evaltypes = ['Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-1-1-1-1',
                 'Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-0.75-1.25-1.25-1.25',
                 'Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-0.5-1.5-1.5-1.5']

if opt.dataset == 'online_products':
    opt.diva_moco_n_key_batches = 70


"""==================================================================================================="""
### The following setting is useful when logging to wandb and running multiple seeds per setup:
### By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
opt.use_tv_split = True
opt.few_shot_evaluate = True
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
    # wandb.config.update(opt, allow_val_change=True)


"""==================================================================================================="""
### Load Remaining Libraries that neeed to be loaded after wandb
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
arch       = 'multifeature_{}'.format(opt.arch)
model      = archs.select(arch, opt)

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


selfsim_model = archs.select(arch, opt)
selfsim_model.load_state_dict(model.state_dict())
_ = selfsim_model.to(opt.device)
_  = model.to(opt.device)


"""============================================================================"""
#################### DATALOADER SETUPS ##################
datasets    = dataset_library.select(opt.dataset, opt, opt.source_path)
if hasattr(model, 'specific_normalization'):
    for dataset_type in datasets.keys():
        if datasets[dataset_type] is not None:
            if not isinstance(datasets[dataset_type], dict):
                datasets[dataset_type].normal_transform_list[-1] = model.specific_normalization
                datasets[dataset_type].normal_transform = torchvision.transforms.Compose(
                    datasets[dataset_type].normal_transform_list
                )
            else:
                for ep_idx in datasets[dataset_type].keys():
                    print("here")
                    datasets[dataset_type][ep_idx]['support'].normal_transform_list[-1] = model.specific_normalization
                    datasets[dataset_type][ep_idx]['support'].normal_transform = torchvision.transforms.Compose(
                        datasets[dataset_type][ep_idx]['support'].normal_transform_list
                    )
                    datasets[dataset_type][ep_idx]['query'].normal_transform_list[-1] = model.specific_normalization
                    datasets[dataset_type][ep_idx]['query'].normal_transform = torchvision.transforms.Compose(
                        datasets[dataset_type][ep_idx]['query'].normal_transform_list
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

datasets['training'].include_aux_augmentations = True
dataloaders['training'] = torch.utils.data.DataLoader(datasets['training'], num_workers=opt.kernels, batch_sampler=train_data_sampler)

opt.n_classes  = len(dataloaders['training'].dataset.avail_classes)
opt.n_test_classes = len(dataloaders['testing'].dataset.avail_classes)


"""============================================================================"""
# Few-Shot episodic dataloaders
episodic_dataloaders = {}
for i, (ep_idx, episode_datasets) in enumerate(datasets['fewshot_episodes'].items()):
    episodic_dataloaders[ep_idx] = {}
    if opt.dataset != 'online_products' and opt.finetune_shots == 1:
        episodic_dataloaders[ep_idx]['support'] = torch.utils.data.DataLoader(episode_datasets['support'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=True)
    else:
        ep_sampler = dsamplers.select(opt.data_sampler, opt, episode_datasets['support'].image_dict, episode_datasets['support'].image_list)
        episodic_dataloaders[ep_idx]['support'] = torch.utils.data.DataLoader(episode_datasets['support'], batch_sampler=ep_sampler, num_workers=opt.kernels)
    episodic_dataloaders[ep_idx]['query'] = torch.utils.data.DataLoader(episode_datasets['query'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)


"""============================================================================"""
#################### CREATE LOGGING FILES ###############
sub_loggers = ['Train', 'Test', 'Model Grad']
if opt.use_tv_split: sub_loggers.append('Val')

if start_from_checkpoint:
    LOG = chkp_data['progress']
else:
    LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)


"""============================================================================"""
#################### LOSS SETUP ####################
batchminer   = bmine.select(opt.batch_mining, opt)
criterion_dict = {}

criterion_dict['discriminative'], to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
opt.diva_decorrnet_lr = opt.lr
criterion_dict['separation'],     to_optim  = criteria.select('adversarial_separation', opt, to_optim, None)
criterion_dict['selfsimilarity'], to_optim  = criteria.select('fast_moco', opt, to_optim, None)
random_shared_batchminer = bmine.select('random_distance', opt)
criterion_dict['shared'], to_optim = criteria.select(opt.loss, opt, to_optim, random_shared_batchminer)
random_intra_batchminer = bmine.select('random_distance', opt)
criterion_dict['intra'], to_optim = criteria.select(opt.loss, opt, to_optim, random_intra_batchminer)

if start_from_checkpoint:
    for key,item in chkp_data['aux']['criterion'][-1].items():
        _ = criterion_dict[key].load_state_dict(item)
    criterion_dict['selfsimilarity'].memory_queue = chkp_data['aux']['criterion'][0]['selfsimilarity'].memory_queue
else:
    criterion_dict['selfsimilarity'].create_memory_queue(selfsim_model, dataloaders['training'], opt.device, opt_key='selfsimilarity')

for key in criterion_dict.keys():
    _ = criterion_dict[key].to(opt.device)

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
miner_text = 'Batchminer:\t {}'.format(opt.batch_mining if criterion_dict['discriminative'].REQUIRES_BATCHMINER else 'N/A')
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

best_val_weights = model.state_dict()
best_val_score = 0
select_epoch = 0

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
        class_labels, input, input_indices, aux_input, imrot_labels = out

        input      = input.to(opt.device)
        model_args = {'x':input.to(opt.device), 'warmup':epoch<opt.warmup}

        out_dict  = model(**model_args)
        embeds, avg_features, features, extra_embeds = [out_dict[key] for key in ['embeds', 'avg_features', 'features', 'extra_embeds']]

        ###
        with torch.no_grad():
            ### Use shuffleBN to avoid information bleeding making samples interdependent.
            forward_shuffle, backward_reorder = criterion_dict['selfsimilarity'].shuffleBN(len(embeds['selfsimilarity']))
            selfsim_key_embeds                = selfsim_model(aux_input[forward_shuffle].to(opt.device))['embeds']
            selfsim_key_embeds                = selfsim_key_embeds['selfsimilarity'][backward_reorder]

        ##########
        loss_args['input_batch']    = input
        loss_args['extra_batches']  = extra_embeds
        loss_args['labels']         = class_labels
        loss_args['f_embed']        = model.model.last_linear
        loss_args['batch_features'] = features
        loss_args['model']          = model
        loss_args['optim']          = optimizer
        loss_args['avg_batch_features'] = avg_features

        ##########
        loss_args['batch'], loss_args['labels'] = embeds['discriminative'], class_labels
        loss_discr                              = criterion_dict['discriminative'](**loss_args)
        loss_args['batch'], loss_args['labels'] = embeds['selfsimilarity'], selfsim_key_embeds
        loss_selfsim                            = criterion_dict['selfsimilarity'](embeds['selfsimilarity'], selfsim_key_embeds)
        loss_args['batch'], loss_args['labels'] = embeds['shared'], class_labels
        loss_shared                             = criterion_dict['shared'](embeds['shared'], class_labels)
        loss_args['batch'], loss_args['labels'] = embeds['intra'], class_labels
        loss_intra                              = criterion_dict['intra'](embeds['intra'], class_labels)
        loss_adv = criterion_dict['separation'](embeds)

        ##########
        loss = loss_discr + opt.diva_alpha_ssl*loss_selfsim + opt.diva_alpha_shared*loss_shared + opt.diva_alpha_intra*loss_intra + loss_adv

        ##########
        optimizer.zero_grad()
        loss.backward()
        loss_collect.append(loss.item())

        ### Compute Model Gradients and log them!
        grads = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
        LOG.progress_saver['Model Grad'].log('Grad L2',  grad_l2,  group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

        ### Update network weights!
        optimizer.step()

        ###
        for model_par, key_model_par in zip(model.parameters(), selfsim_model.parameters()):
            momentum = criterion_dict['selfsimilarity'].momentum
            key_model_par.data.copy_(key_model_par.data*momentum + model_par.data*(1-momentum))
        criterion_dict['selfsimilarity'].update_memory_queue(selfsim_key_embeds)

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

    """======================================="""
    ### Evaluate Metric for Training & Test (& Validation)
    _ = model.eval()

    aux_store = {'optimizer':optimizer.state_dict(), 'criterion':(criterion_dict, {key:criterion.state_dict() for key,criterion in criterion_dict.items()}), 'scheduler':scheduler.state_dict(), 'dataloaders':dataloaders,
                 'datasets':datasets, 'train_data_sampler':train_data_sampler}

    if opt.use_tv_split:
        print('\nComputing Validation Metrics...')
        eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['validation']], model, opt, opt.evaltypes, opt.device, log_key='Val', aux_store=aux_store)

        if opt.dataset == 'cub200':
            typename = 'Combined_discriminative_shared_selfsimilarity_intra-0.75-1.25-1.25-1.25'
        elif opt.dataset == 'cars196':
            typename = 'Combined_discriminative_shared_selfsimilarity_intra-0.5-1.5-1.5-1.5'
        elif opt.datset == 'online_products':
            typename = 'Combined_discriminative_shared_selfsimilarity_intra-1-1-1-1'

        val_recall_score = LOG.progress_saver['Val'].groups['{}_e_recall'.format(typename)]['e_recall@1']['content'][-1]
        val_map_score = LOG.progress_saver['Val'].groups['{}_mAP_1000'.format(typename)]['mAP_1000']['content'][-1]
        if opt.recall_only:
            total_val_score = val_recall_score
        else:
            total_val_score = val_recall_score + val_map_score
        if total_val_score > best_val_score:
            best_val_score = total_val_score
            best_weights = model.state_dict()
            select_epoch = epoch

    if not opt.no_train_metrics:
        print('\nComputing Training Metrics...')
        eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['evaluation']], model, opt, opt.evaltypes, opt.device, log_key='Train', aux_store=aux_store)

    import wandb
    LOG.update(all=True)

    eval.set_checkpoint(model, opt, LOG, LOG.prop.save_path+'/checkpoint.pth.tar', aux=aux_store)


    """======================================="""
    print('\nTotal Epoch Runtime: {0:4.2f}s'.format(time.time()-epoch_start_time))
    print('\n-----\n')


"""======================================================="""
opt.completed = True
pkl.dump(opt,open(opt.save_path+"/hypa.pkl","wb"))


"""====== Evalute Few-Shot Performance for best DML model. ==========="""
_ = model.load_state_dict(best_weights)
_ = model.eval()

import utilities.finetune_utils as f_utils


finetune_params = {'optim': opt.finetune_optim, 'only_last': opt.finetune_only_last,
                   'criterion': opt.finetune_criterion, 'lr': opt.finetune_lr_multi * opt.lr,
                   'iter': opt.finetune_iter}

few_shot_ep_metrics_coll = {}
zero_shot_ep_metrics_coll = {}

print('Computing few-shot results...')
for ep_idx in range(len(episodic_dataloaders)):
    print('--- Episode {}. ---'.format(ep_idx + 1))
    support_dataloader = episodic_dataloaders[ep_idx]['support']
    query_dataloader = episodic_dataloaders[ep_idx]['query']

    opt.n_classes = len(support_dataloader.dataset.avail_classes)

    zero_shot_ep_metrics = eval.evaluate(opt.dataset, LOG, metric_computer, [query_dataloader], model, opt, opt.evaltypes, opt.device, compute_metrics_only=True, print_text=False)
    model_copy = copy.deepcopy(model)
    _ = model_copy.train()
    f_utils.finetune_model(opt, model_copy, support_dataloader, opt.device, finetune_params, reweight=opt.finetune_reweight)
    _ = model_copy.eval()
    few_shot_ep_metrics = eval.evaluate(opt.dataset, LOG, metric_computer, [query_dataloader], model_copy, opt, opt.evaltypes, opt.device, compute_metrics_only=True, print_text=False)

    for key, item in few_shot_ep_metrics[typename].items():
        if key not in few_shot_ep_metrics_coll:
            few_shot_ep_metrics_coll[key] = []
        few_shot_ep_metrics_coll[key].append(item)

    for key, item in zero_shot_ep_metrics[typename].items():
        if key not in zero_shot_ep_metrics_coll:
            zero_shot_ep_metrics_coll[key] = []
        zero_shot_ep_metrics_coll[key].append(item)

import scipy
few_res = {}
zero_res = {}
confidence = 0.95
few_shot_ep_metrics_coll = {key: (np.mean(item), scipy.stats.sem(item) * scipy.stats.t._ppf((1+confidence)/2., len(item)-1)) for key, item in few_shot_ep_metrics_coll.items()}
zero_shot_ep_metrics_coll = {key: (np.mean(item), scipy.stats.sem(item) * scipy.stats.t._ppf((1+confidence)/2., len(item)-1)) for key, item in zero_shot_ep_metrics_coll.items()}

if opt.log_online:
    summary_table = wandb.Table(columns=["Metric", "Few-Shot Mean", "Few-Shot Conf.", "Zero-Shot Mean", "Zero-Shot Conf."])
    for (few_key, few_item), (zero_key, zero_item) in zip(few_shot_ep_metrics_coll.items(), zero_shot_ep_metrics_coll.items()):
        summary_table.add_data(few_key, *few_item, *zero_item)
    wandb.log({"Test_Results": summary_table}, commit=False)
    wandb.log({'few_shot_eval_{}_mean'.format(key): item[0] for key, item in few_shot_ep_metrics_coll.items()}, commit=False)
    wandb.log({'few_shot_eval_{}_conf'.format(key): item[1] for key, item in few_shot_ep_metrics_coll.items()}, commit=False)
    wandb.log({'zero_shot_eval_{}_mean'.format(key): item[0] for key, item in zero_shot_ep_metrics_coll.items()}, commit=False)
    wandb.log({'zero_shot_eval_{}_conf'.format(key): item[1] for key, item in zero_shot_ep_metrics_coll.items()}, commit=False)

summary_table = [["Metric", "Few-Shot Mean", "Few-Shot Conf.", "Zero-Shot Mean", "Zero-Shot Conf."]]
for (few_key, few_item), (zero_key, zero_item) in zip(few_shot_ep_metrics_coll.items(), zero_shot_ep_metrics_coll.items()):
    summary_table.append([few_key, *few_item, *zero_item])
summary_table = np.array(summary_table)
pd.DataFrame(summary_table[1:], columns=summary_table[0]).to_csv(opt.save_path+'/test_summary.txt', index=None)
