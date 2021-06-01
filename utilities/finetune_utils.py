from utilities import misc
import numpy as np
import random
import tqdm
import torch
import criteria as criteria
import batchminer as bmine

def finetune_model(opt, model, dataloader, device, finetune_params, seed=None, reweight=True):
    if seed is not None:
        misc.set_seed(seed)
    else:
        misc.set_seed(opt.seed)


    if finetune_params['optim'] == 'sgd':
        optim_f = torch.optim.SGD
    elif finetune_params['optim'] == 'adam':
        optim_f = torch.optim.Adam

    if opt.dataset == 'cub200':
        weights = [0.75, 1.25, 1.25, 1.25]
    elif opt.dataset == 'cars196':
        weights = [0.5, 1.5, 1.5, 1.5]
    elif opt.datset == 'online_products':
        weights = [1., 1., 1., 1.]

    if finetune_params['only_last']:
        # if 'multifeature' in model.name and opt.optim_weights:
        #     weights = torch.nn.Parameter(torch.Tensor(weights))
        #     to_optim = [{'params': model.model.last_linear.parameters(), 'lr': finetune_params['lr']},
        #                 {'params': weights, 'lr': finetune_params['lr']}]
        #     finetune_optim = optim_f(to_optim)
        #     _ = weights.to(opt.device)
        # else:
        finetune_optim = optim_f(model.model.last_linear.parameters(), lr=finetune_params['lr'])

        _ = model.eval()
        _ = model.model.last_linear.train()
    else:
        finetune_optim = optim_f(model.parameters(), lr=finetune_params['lr'])
        _ = model.train()

    # finetune_dataset.image_dict
    batchminer = bmine.select('random', opt)
    if finetune_params['criterion'] == 'multisimilarity':
        if 'margin' in opt.loss:
            opt.loss_multisimilarity_d_mode = 'euclidean'
        criterion, _ = criteria.select('multisimilarity', opt, [], batchminer)
    elif finetune_params['criterion'] == 'margin':
        batchminer = bmine.select('distance', opt)
        criterion, _ = criteria.select('margin', opt, [], batchminer)
    elif finetune_params['criterion'] == 'triplet':
        criterion, _ = criteria.select('triplet', opt, [], batchminer)


    loss_collect = []
    finetune_iterator = tqdm.tqdm(range(finetune_params['iter']), total=finetune_params['iter'], desc='Finetuning...')
    # finetune_iterator = tqdm.tqdm(range(finetune_params['iter']), total=np.clip(int(np.ceil(finetune_params['iter']/len(dataloader)))-1, 1, None), desc='Finetuning...')
    count = 0
    for i in range(finetune_params['iter']):
        for inp in dataloader:
            input_img, target = inp[1], inp[0]
            out_dict = model(input_img.to(device), warmup=finetune_params['only_last'])
            if 'multifeature' in model.name:
                if reweight:
                    weighted_subfeatures = [weights[i]*out_dict['embeds'][subevaltype] for i,subevaltype in enumerate(['discriminative', 'shared', 'selfsimilarity', 'intra'])]
                else:
                    weighted_subfeatures = [out_dict['embeds'][subevaltype] for i, subevaltype in enumerate(['discriminative', 'shared', 'selfsimilarity', 'intra'])]
                if 'normalize' in model.name:
                    out_dict['embeds'] = torch.nn.functional.normalize(torch.cat(weighted_subfeatures, dim=-1), dim=-1)
                else:
                    out_dict['embeds'] = torch.cat(weighted_subfeatures, dim=-1)

            loss_args = {'batch': out_dict['embeds'], 'labels': target}
            # loss_args = {'batch': out_dict['embeds'].to(opt.device), 'labels': target.to(opt.device)}
            finetune_optim.zero_grad()
            loss = criterion(**loss_args)
            loss.backward()
            loss_collect.append(loss.item())
            finetune_optim.step()
            finetune_iterator.set_postfix_str('Loss: {0:3.5f}'.format(np.mean(loss_collect)))

            count += 1

            finetune_iterator.update(1)
            if count == finetune_params['iter']:
                break

        if count == finetune_params['iter']:
            break






def nonredundant_finetuner(opt, backbone, dataloader, device,
                           finetune_lr, finetune_iter, finetune_criterion='margin', finetune_optim='adam',
                           head=None, seed=None, optim_head_only=False, ):
    if seed is not None:
        misc.set_seed(seed)
    else:
        misc.set_seed(opt.seed)

    if finetune_optim == 'sgd':
        optim_f = torch.optim.SGD
    elif finetune_optim == 'adam':
        optim_f = torch.optim.Adam

    if optim_head_only:
        finetune_optim = optim_f(head.parameters(), lr=finetune_lr)
        _ = backbone.eval()
        _ = head.train()
    else:
        finetune_optim = optim_f(
            [{'params': backbone.parameters()},
             {'params': head.parameters()}], lr=finetune_lr
            )
        _ = backbone.train()
        _ = head.train()

    # finetune_dataset.image_dict
    batchminer = bmine.select('random', opt)
    if finetune_criterion == 'multisimilarity':
        if 'margin' in opt.loss:
            opt.loss_multisimilarity_d_mode = 'euclidean'
        criterion, _ = criteria.select('multisimilarity', opt, [], batchminer)
    elif finetune_criterion == 'margin':
        batchminer = bmine.select('distance', opt)
        criterion, _ = criteria.select('margin', opt, [], batchminer)
    elif finetune_criterion == 'triplet':
        criterion, _ = criteria.select('triplet', opt, [], batchminer)


    loss_collect = []
    finetune_iterator = tqdm.tqdm(range(finetune_iter), total=finetune_iter, desc='Finetuning...')
    # finetune_iterator = tqdm.tqdm(range(finetune_params['iter']), total=np.clip(int(np.ceil(finetune_params['iter']/len(dataloader)))-1, 1, None), desc='Finetuning...')
    count = 0
    for i in range(finetune_iter):
        for inp in dataloader:
            input_img, target = inp[1], inp[0]
            if optim_head_only:
                with torch.no_grad():
                    out = backbone(input_img.to(device))
            else:
                out = backbone(input_img.to(device))
            if isinstance(out, dict):
                out = out['avg_features']
            if head is not None:
                out = head(out.to(torch.float).to(device))

            loss_args = {'batch': out, 'labels': target}
            # loss_args = {'batch': out_dict['embeds'].to(opt.device), 'labels': target.to(opt.device)}
            finetune_optim.zero_grad()
            loss = criterion(**loss_args)
            loss.backward()
            loss_collect.append(loss.item())
            finetune_optim.step()
            finetune_iterator.set_postfix_str('Loss: {0:3.5f}'.format(np.mean(loss_collect)))

            count += 1

            finetune_iterator.update(1)
            if count == finetune_iter:
                break

        if count == finetune_iter:
            break
