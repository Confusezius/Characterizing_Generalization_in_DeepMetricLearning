from datasets.basic_dataset_scaffold import BaseDataset
import os, copy

def give_info_dict(source, classes):
    image_list = [[(i,source + '/' + classname +'/'+x) for x in sorted(os.listdir(source + '/' + classname)) if '._' not in x] for i,classname in enumerate(classes)]
    image_list = [x for y in image_list for x in y]

    idx_to_class_conversion = {i:classname for i,classname in enumerate(classes)}

    image_dict = {}
    for key,img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    return image_list, image_dict, idx_to_class_conversion


def OODatasets(opt, datapath, splitpath=None):
    import pickle as pkl
    splitpath_base = os.getcwd() if splitpath is None else splitpath
    split_dict = pkl.load(open(splitpath_base+'/datasplits/cub200_splits.pkl', 'rb'))
    train_classes, test_classes, fid = split_dict[opt.data_hardness]['train'], split_dict[opt.data_hardness]['test'], split_dict[opt.data_hardness]['fid']
    print('\nLoaded Data Split with FID-Hardness: {0:4.4f}'.format(fid))

    ###
    image_sourcepath = datapath + '/images'

    ###
    if opt.use_tv_split:
        if not opt.tv_split_perc:
            train_classes, val_classes = split_dict[opt.data_hardness]['split_train'], split_dict[opt.data_hardness]['split_val']
        else:
            train_val_split = int(len(train_classes)*opt.tv_split_perc)
            train_classes, val_classes = train_classes[:train_val_split], train_classes[train_val_split:]
        val_image_list, val_image_dict, val_conversion = give_info_dict(image_sourcepath, val_classes)
        val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
        val_dataset.conversion = val_conversion
    else:
        val_dataset, val_image_dict = None, None

    ###
    train_image_list, train_image_dict, train_conversion = give_info_dict(image_sourcepath, train_classes)
    test_image_list, test_image_dict, test_conversion    = give_info_dict(image_sourcepath, test_classes)

    ###
    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict is not None else 'X', len(test_image_dict)))

    ###
    train_dataset      = BaseDataset(train_image_dict, opt)
    test_dataset       = BaseDataset(test_image_dict,  opt, is_validation=True)
    train_eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)

    ###
    reverse_train_conversion = {item: key for key, item in train_conversion.items()}
    reverse_test_conversion = {item: key for key, item in test_conversion.items()}

    train_dataset.conversion       = train_conversion
    test_dataset.conversion        = test_conversion
    train_eval_dataset.conversion  = train_conversion

    ###
    few_shot_datasets = None
    episode_context = None
    if hasattr(opt, 'few_shot_evaluate'):
        test_episodes = split_dict[opt.data_hardness]['test_episodes']
        shots = list(test_episodes.keys())
        episode_idxs = list(test_episodes[opt.finetune_shots].keys())
        classnames = list(test_episodes[opt.finetune_shots][episode_idxs[0]].keys())
        conv_classnames = [reverse_test_conversion[classname] for classname in classnames]

        episode_context = {}
        for ep_idx in episode_idxs:
            ref_dict = copy.deepcopy(test_image_dict)
            test_support_image_dict = {}
            test_query_image_dict = {}
            for conv_classname, classname in zip(conv_classnames, classnames):
                samples_to_use = test_episodes[opt.finetune_shots][ep_idx][classname]
                base_path = '/'.join(ref_dict[conv_classname][0].split('/')[:-1])
                support_samples_to_use = [base_path + '/' + x for x in samples_to_use]
                query_samples_to_use = [x for x in ref_dict[conv_classname] if x not in support_samples_to_use]
                test_query_image_dict[conv_classname] = query_samples_to_use
                test_support_image_dict[conv_classname] = support_samples_to_use

            test_support_dataset = BaseDataset(test_support_image_dict, opt)
            test_query_dataset = BaseDataset(test_query_image_dict, opt, is_validation=True)

            episode_context[ep_idx] = {'support': test_support_dataset, 'query': test_query_dataset}

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':train_eval_dataset, 'fewshot_episodes': episode_context}





def DefaultDatasets(opt, datapath):
    image_sourcepath  = datapath+'/images'
    image_classes     = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0]))
    total_conversion  = {int(x.split('.')[0])-1:x.split('.')[-1] for x in image_classes}
    image_list        = {int(key.split('.')[0])-1:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes}
    image_list        = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list        = [x for y in image_list for x in y]

    ### Dictionary of structure class:list_of_samples_with_said_class
    image_dict    = {}
    for key, img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    ### Use the first half of the sorted data as training and the second half as test set
    keys = sorted(list(image_dict.keys()))

    train,test = keys[:len(keys)//2], keys[len(keys)//2:]

    ### If required, split the training data into a train/val setup either by or per class.
    # from IPython import embed; embed()
    if opt.use_tv_split:
        if not opt.tv_split_by_samples:
            train_val_split = int(len(train)*opt.tv_split_perc)
            train, val      = train[:train_val_split], train[train_val_split:]
            ###
            train_image_dict = {i:image_dict[key] for i,key in enumerate(train)}
            val_image_dict   = {i:image_dict[key] for i,key in enumerate(val)}
            test_image_dict  = {i:image_dict[key] for i,key in enumerate(test)}
        else:
            val = train
            train_image_dict, val_image_dict = {},{}
            for key in train:
                train_ixs   = np.array(list(set(np.round(np.linspace(0,len(image_dict[key])-1,int(len(image_dict[key])*opt.tv_split_perc)))))).astype(int)
                val_ixs     = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key]   = np.array(image_dict[key])[val_ixs]
        val_dataset    = BaseDataset(val_image_dict, opt, is_validation=True)
        val_conversion = {i:total_conversion[key] for i,key in enumerate(val)}
        ###
        val_dataset.conversion   = val_conversion
    else:
        train_image_dict = {key:image_dict[key] for key in train}
        val_image_dict   = None
        val_dataset      = None

    ###
    train_conversion = {i:total_conversion[key] for i,key in enumerate(train)}
    test_conversion  = {i:total_conversion[key] for i,key in enumerate(test)}

    ###
    test_image_dict = {key:image_dict[key] for key in test}

    ###
    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    ###
    train_dataset       = BaseDataset(train_image_dict, opt)
    test_dataset        = BaseDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset        = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)
    train_dataset.conversion       = train_conversion
    test_dataset.conversion        = test_conversion
    eval_dataset.conversion        = test_conversion
    eval_train_dataset.conversion  = train_conversion


    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}
