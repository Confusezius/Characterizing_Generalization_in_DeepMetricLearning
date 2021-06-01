from datasets.basic_dataset_scaffold import BaseDataset
import os, numpy as np, copy
import pandas as pd




def OODatasets(opt, datapath=None, splitpath=None):
    import pickle as pkl
    splitpath_base = os.getcwd() if splitpath is None else splitpath
    split_dict = pkl.load(open(splitpath_base+'/datasplits/online_products_splits.pkl', 'rb'))
    train_classes, test_classes, fid = split_dict[opt.data_hardness]['train'], split_dict[opt.data_hardness]['test'], split_dict[opt.data_hardness]['fid']
    print('\nLoaded Data Split with FID-Hardness: {0:4.4f}'.format(fid))


    image_sourcepath  = opt.source_path+'/images'
    training_files = pd.read_table(opt.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(opt.source_path+'/Info_Files/Ebay_test.txt',  header=0, delimiter=' ')

    super_dict       = {}
    super_conversion = {}
    class_dict       = {}
    class_conversion = {}

    for i,(super_ix, class_ix, image_path) in enumerate(zip(training_files['super_class_id'],training_files['class_id'],training_files['path'])):
        if super_ix not in super_dict: super_dict[super_ix] = {}
        if class_ix not in super_dict[super_ix]: super_dict[super_ix][class_ix] = []
        super_dict[super_ix][class_ix].append(image_sourcepath+'/'+image_path)

        if class_ix not in class_dict: class_dict[class_ix] = []
        class_dict[class_ix].append(image_sourcepath + '/' + image_path)
        class_conversion[class_ix] = image_path.split('/')[-1].split('_')[0]

    for i,(super_ix, class_ix, image_path) in enumerate(zip(test_files['super_class_id'],test_files['class_id'],test_files['path'])):
        if super_ix not in super_dict: super_dict[super_ix] = {}
        if class_ix not in super_dict[super_ix]: super_dict[super_ix][class_ix] = []
        super_dict[super_ix][class_ix].append(image_sourcepath+'/'+image_path)

        if class_ix not in class_dict: class_dict[class_ix] = []
        class_dict[class_ix].append(image_sourcepath + '/' + image_path)
        class_conversion[class_ix] = image_path.split('/')[-1].split('_')[0]

    train_image_dict = {key:item for key,item in class_dict.items() if str(class_conversion[key]) in train_classes}
    test_image_dict  = {key:item for key,item in class_dict.items() if str(class_conversion[key]) in test_classes}

    val_conversion = None
    if opt.use_tv_split:
        if not opt.tv_split_perc:
            train_classes, val_classes = split_dict[opt.data_hardness]['split_train'], split_dict[opt.data_hardness]['split_val']
        else:
            train_val_split_class      = int(len(train_image_dict)*opt.tv_split_perc)
            train_classes, val_classes = np.array(list(train_image_dict.keys()))[:train_val_split_class], np.array(list(train_image_dict.keys()))[train_val_split_class:]
        train_image_dict = {key:item for key,item in class_dict.items() if key in train_classes}
        val_image_dict   = {key:item for key,item in class_dict.items() if key in val_classes}
    else:
        val_image_dict   = None

    train_classes, test_classes = sorted(list(train_image_dict.keys())), sorted(list(test_image_dict.keys()))
    train_conversion = {i: classname for i, classname in enumerate(train_classes)}
    test_conversion = {i: classname for i, classname in enumerate(test_classes)}

    train_image_dict = {i:train_image_dict[key] for i,key in enumerate(train_classes)}
    test_image_dict  = {i:test_image_dict[key] for i,key in enumerate(test_classes)}
    if opt.use_tv_split:
        val_classes      = sorted(list(val_image_dict.keys()))
        val_image_dict   = {i:val_image_dict[key] for i,key in enumerate(val_classes)}
        val_conversion = {i: classname for i, classname in enumerate(val_classes)}
        reverse_val_conversion = {item: key for key, item in val_conversion.items()}

    ##
    if val_image_dict:
        val_dataset            = BaseDataset(val_image_dict,   opt, is_validation=True)
        val_dataset.conversion = val_conversion
    else:
        val_dataset = None

    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    train_dataset       = BaseDataset(train_image_dict, opt)
    eval_dataset        = BaseDataset(train_image_dict, opt, is_validation=True)
    test_dataset        = BaseDataset(test_image_dict,  opt, is_validation=True)

    # super_train_dataset.conversion = super_train_conversion
    reverse_train_conversion = {item: key for key, item in train_conversion.items()}
    reverse_test_conversion = {item: key for key, item in test_conversion.items()}

    train_dataset.conversion       = train_conversion
    eval_dataset.conversion        = train_conversion
    test_dataset.conversion        = test_conversion

    few_shot_datasets = None
    episode_context = None
    if hasattr(opt, 'few_shot_evaluate'):
        test_episodes = split_dict[opt.data_hardness]['test_episodes']
        shots = list(test_episodes.keys())
        episode_idxs = list(test_episodes[opt.finetune_shots].keys())

        episode_context = {}
        for ep_idx in episode_idxs:
            ref_dict = copy.deepcopy(test_image_dict)
            test_support_image_dict = {}
            test_query_image_dict = {}

            classes_to_use = test_episodes[opt.finetune_shots][ep_idx]
            classes_to_use = [reverse_test_conversion[x] for x in classes_to_use]
            test_support_image_dict = {classname:test_image_dict[classname] for classname in classes_to_use}
            test_query_image_dict = {classname:test_image_dict[classname] for classname in test_image_dict.keys() if classname not in test_support_image_dict}

            test_support_dataset = BaseDataset(test_support_image_dict, opt)
            test_query_dataset = BaseDataset(test_query_image_dict, opt, is_validation=True)

            episode_context[ep_idx] = {'support': test_support_dataset, 'query': test_query_dataset}


    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'fewshot_episodes': episode_context, 'evaluation_train':None, 'super_evaluation':None}



def DefaultDatasets(opt, datapath=None):
    image_sourcepath  = opt.source_path+'/images'
    training_files = pd.read_table(opt.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(opt.source_path+'/Info_Files/Ebay_test.txt', header=0, delimiter=' ')

    spi   = np.array([(a,b) for a,b in zip(training_files['super_class_id'], training_files['class_id'])])
    super_dict       = {}
    super_conversion = {}
    for i,(super_ix, class_ix, image_path) in enumerate(zip(training_files['super_class_id'],training_files['class_id'],training_files['path'])):
        if super_ix not in super_dict: super_dict[super_ix] = {}
        if class_ix not in super_dict[super_ix]: super_dict[super_ix][class_ix] = []
        super_dict[super_ix][class_ix].append(image_sourcepath+'/'+image_path)

    if opt.use_tv_split:
        if not opt.tv_split_by_samples:
            train_image_dict, val_image_dict = {},{}
            train_count, val_count = 0, 0
            for super_ix in super_dict.keys():
                class_ixs       = sorted(list(super_dict[super_ix].keys()))
                train_val_split = int(len(super_dict[super_ix])*opt.tv_split_perc)
                train_image_dict[super_ix] = {}
                for _,class_ix in enumerate(class_ixs[:train_val_split]):
                    train_image_dict[super_ix][train_count] = super_dict[super_ix][class_ix]
                    train_count += 1
                val_image_dict[super_ix] = {}
                for _,class_ix in enumerate(class_ixs[train_val_split:]):
                    val_image_dict[super_ix][val_count]     = super_dict[super_ix][class_ix]
                    val_count += 1
        else:
            train_image_dict, val_image_dict = {},{}
            for super_ix in super_dict.keys():
                class_ixs       = sorted(list(super_dict[super_ix].keys()))
                train_image_dict[super_ix] = {}
                val_image_dict[super_ix]   = {}
                for class_ix in class_ixs:
                    train_val_split = int(len(super_dict[super_ix][class_ix])*opt.tv_split_perc)
                    train_image_dict[super_ix][class_ix] = super_dict[super_ix][class_ix][:train_val_split]
                    val_image_dict[super_ix][class_ix]   = super_dict[super_ix][class_ix][train_val_split:]
    else:
        train_image_dict = super_dict
        val_image_dict   = None

    ####
    test_image_dict        = {}
    train_image_dict_temp  = {}
    val_image_dict_temp    = {}
    super_train_image_dict = {}
    super_val_image_dict   = {}
    train_conversion       = {}
    super_train_conversion = {}
    val_conversion         = {}
    super_val_conversion   = {}
    test_conversion        = {}
    super_test_conversion  = {}

    ## Create Training Dictionaries
    i = 0
    for super_ix,super_set in train_image_dict.items():
        super_ix -= 1
        counter   = 0
        super_train_image_dict[super_ix] = []
        for class_ix,class_set in super_set.items():
            class_ix -= 1
            super_train_image_dict[super_ix].extend(class_set)
            train_image_dict_temp[class_ix] = class_set
            if class_ix not in train_conversion:
                train_conversion[class_ix] = class_set[0].split('/')[-1].split('_')[0]
                super_conversion[class_ix] = class_set[0].split('/')[-2]
            counter += 1
            i       += 1
    train_image_dict = train_image_dict_temp

    ## Create Validation Dictionaries
    if opt.use_tv_split:
        i = 0
        for super_ix,super_set in val_image_dict.items():
            super_ix -= 1
            counter   = 0
            super_val_image_dict[super_ix] = []
            for class_ix,class_set in super_set.items():
                class_ix -= 1
                super_val_image_dict[super_ix].extend(class_set)
                val_image_dict_temp[class_ix] = class_set
                if class_ix not in val_conversion:
                    val_conversion[class_ix] = class_set[0].split('/')[-1].split('_')[0]
                    super_conversion[class_ix] = class_set[0].split('/')[-2]
                counter += 1
                i       += 1
        val_image_dict = val_image_dict_temp
    else:
        val_image_dict = None

    ## Create Test Dictioniaries
    for class_ix, img_path in zip(test_files['class_id'],test_files['path']):
        class_ix = class_ix-1
        if not class_ix in test_image_dict.keys():
            test_image_dict[class_ix] = []
        test_image_dict[class_ix].append(image_sourcepath+'/'+img_path)
        test_conversion[class_ix]       = img_path.split('/')[-1].split('_')[0]
        super_test_conversion[class_ix] = img_path.split('/')[-2]

    ##
    if val_image_dict:
        val_dataset            = BaseDataset(val_image_dict,   opt, is_validation=True)
        val_dataset.conversion = val_conversion
    else:
        val_dataset = None

    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    super_train_dataset = BaseDataset(super_train_image_dict, opt, is_validation=True)
    train_dataset       = BaseDataset(train_image_dict, opt)
    test_dataset        = BaseDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset        = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt)

    super_train_dataset.conversion = super_train_conversion
    train_dataset.conversion       = train_conversion
    test_dataset.conversion        = test_conversion
    eval_dataset.conversion        = train_conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset, 'super_evaluation':super_train_dataset}
