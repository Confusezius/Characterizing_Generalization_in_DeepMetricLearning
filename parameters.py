import argparse, os


#######################################
def basic_training_parameters(parser):
    parser.add_argument('--dataset',
                        default='cub200',
                        type=str,
                        help='Dataset to use.')
    parser.add_argument(
        '--use_tv_split',
        action='store_true',
        help=
        'Flag: split data into training/validation (by classes).'
    )
    parser.add_argument(
        '--tv_split_by_samples',
        action='store_true',
        help=
        'Flag: split data into training/validation (by sample count).'
    )
    parser.add_argument(
        '--tv_split_perc',
        default=0,
        type=float,
        help=
        'Percentage with which the training dataset is split into training/validation.'
    )
    parser.add_argument(
        '--checkpoint',
        action='store_true',
        help=
        'Flag: Use checkpointing so training can be continued if interrupted (on a per-epoch-basis).'
    )
    parser.add_argument(
        '--completed',
        action='store_true',
        help=
        'Flag: Simply highlights if a training run is completed and can be skipped. Primarily used internally.'
    )
    parser.add_argument(
        '--dont_train_eval',
        action='store_true',
        help=
        'Flag: Dont evalute on training data.'
    )
    parser.add_argument(
        '--no_train_metrics',
        action='store_true',
        help=
        'Flag: Dont compute metrics on training data.'
    )

    ### General Training Parameters
    parser.add_argument('--lr',
                        default=0.00001,
                        type=float,
                        help='Learning Rate for network parameters.')
    parser.add_argument('--fc_lr',
                        default=-1,
                        type=float,
                        help='Learning Rate for fully-connect, last layer parameters.')
    parser.add_argument('--n_epochs',
                        default=150,
                        type=int,
                        help='Number of training epochs.')
    parser.add_argument('--kernels',
                        default=6,
                        type=int,
                        help='Number of workers for pytorch dataloader.')
    parser.add_argument('--bs',
                        default=112,
                        type=int,
                        help='Mini-Batchsize to use.')
    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument(
        '--scheduler',
        default='step',
        type=str,
        help='Type of learning rate scheduling. Currently: step & exp.')
    parser.add_argument('--gamma',
                        default=0.3,
                        type=float,
                        help='Learning rate reduction after tau epochs.')
    parser.add_argument('--decay',
                        default=0.0004,
                        type=float,
                        help='Weight decay for optimizer.')
    parser.add_argument('--tau',
                        default=[10000],
                        nargs='+',
                        type=int,
                        help='Stepsize before reducing learning rate.')
    parser.add_argument(
        '--augmentation',
        default='base',
        type=str,
        help='Type of data augmentation mode. Default used DML default.')
    parser.add_argument(
        '--warmup',
        default=0,
        type=int,
        help=
        'Number of warmup epochs where backbone is frozen and only the last layer is trained.'
    )

    parser.add_argument('--internal_split',
                        default=1,
                        type=float,
                        help='Internal split value used for minibatch-construction. Left to default.')
    parser.add_argument(
        '--evaluate_on_cpu',
        action='store_true',
        help=
        'Flag: Evaluates metrics solely on CPU.'
    )
    parser.add_argument(
        '--load_to_ram',
        action='store_true',
        help=
        'Flag: If enough ram is available, load all data to RAM.'
    )


    ##### Loss-specific Settings
    parser.add_argument('--optim',
                        default='adam',
                        type=str,
                        help='Optimizer to use. Currently uses ADAM.')
    parser.add_argument('--loss',
                        default='margin',
                        type=str,
                        help='DML training criterion to use. See "criteria/__init__.py" for options.')
    parser.add_argument(
        '--batch_mining',
        default='distance',
        type=str,
        help=
        'Batch-mining method to accompany the DML objective.'
    )

    #####
    parser.add_argument(
        '--embed_dim',
        default=128,
        type=int,
        help=
        'Embedding dimensionality of the network. Note: dim=128 or 64 is used in most papers.'
    )
    parser.add_argument(
        '--arch',
        default='resnet50_frozen_normalize',
        type=str,
        help='Underlying network architecture. Frozen denotes that exisiting pretrained batchnorm layers are frozen, and normalize denotes normalization of the output embedding.'
    )
    parser.add_argument('--not_pretrained', action='store_true')
    parser.add_argument('--no_loss_schedules', action='store_true')

    #####
    parser.add_argument('--evaluation_metrics',
                        nargs='+',
                        default=[
                            'e_recall@1', 'e_recall@2', 'e_recall@4', 'nmi',
                            'f1', 'mAP_1000', 'mAP_c', 'dists@intra',
                            'dists@inter', 'dists@intra_over_inter',
                            'rho_spectrum@0', 'rho_spectrum@-1',
                            'rho_spectrum@1', 'rho_spectrum@2', 'rho_spectrum@10'
                        ],
                        type=str,
                        help='Metrics to evaluate performance by.')
    parser.add_argument(
        '--evaltypes',
        nargs='+',
        default=['embeds'],
        type=str,
        help=
        'The network may produce multiple embeddings (ModuleDict). If the key is listed here, the entry will be evaluated on the evaluation metrics.\
                                                                                                       Note: One may use Combined_embed1_embed2_..._embedn-w1-w1-...-wn to compute evaluation metrics on weighted (normalized) combinations.'
    )
    parser.add_argument(
        '--storage_metrics',
        nargs='+',
        default=['e_recall@1'],
        type=str,
        help=
        'Improvement in these metrics on the test/valset trigger checkpointing.')

    ##### Setup Parameters
    parser.add_argument('--gpu',
                        default=[0],
                        nargs='+',
                        type=int,
                        help='GPU-ID to use.')
    parser.add_argument(
        '--savename',
        default='group_plus_seed',
        type=str,
        help=
        'Save-folder naming string.'
    )
    parser.add_argument('--source_path',
                        default=os.getcwd() + '/../../Datasets',
                        type=str,
                        help='Path to training data.')
    parser.add_argument('--save_path',
                        default=os.getcwd() + '/Training_Results',
                        type=str,
                        help='Where to save everything.')

    return parser


#######################################
def s2sd_parameters(parser):
    #Training Criteria
    parser.add_argument('--loss_s2sd_source',
                        default='multisimilarity',
                        type=str,
                        help='DML criterion for the base embedding branch.')
    parser.add_argument(
        '--loss_s2sd_target',
        default='multisimilarity',
        type=str,
        help='DML criterion for the target embedding branches.')
    #Basic S2SD
    parser.add_argument('--loss_s2sd_T',
                        default=1,
                        type=float,
                        help='Temperature for the KL-Divergence Distillation.')
    parser.add_argument('--loss_s2sd_w',
                        default=50,
                        type=float,
                        help='Weight of the distillation loss.')
    parser.add_argument(
        '--loss_s2sd_pool_aggr',
        action='store_true',
        help=
        'Flag. If set, uses both global max- and average pooling in the target branches.'
    )
    parser.add_argument(
        '--loss_s2sd_target_dims',
        default=[1024, 1536, 2048],
        nargs='+',
        type=int,
        help='Defines number and dimensionality of used target branches.')
    #Feature Space Distillation
    parser.add_argument('--loss_s2sd_feat_distill',
                        action='store_true',
                        help='Flag. If set, feature distillation is used.')
    parser.add_argument('--loss_s2sd_feat_w',
                        default=50,
                        type=float,
                        help='Weight of the feature space distillation loss.')
    parser.add_argument(
        '--loss_s2sd_feat_distill_delay',
        default=1000,
        type=int,
        help=
        'Defines the number of training iterations before feature distillation is activated.'
    )
    return parser


#######################################
def diva_parameters(parser):
    ##### Multifeature Parameters
    parser.add_argument('--diva_ssl',
                        default='fast_moco',
                        type=str,
                        help='Self-supervised Objective to use.')
    parser.add_argument('--diva_sharing',
                        default='random',
                        type=str,
                        help='Objective to use for shared feature mining.')
    parser.add_argument('--diva_intra',
                        default='random',
                        type=str,
                        help='Objective to use for intraclass feature mining.')
    parser.add_argument(
        '--diva_features',
        default=['discriminative', 'shared', 'selfsimilarity', 'intra'],
        nargs='+',
        type=str,
        help='Types of features to mine in DiVA.')
    parser.add_argument('--diva_decorrelations',
                        default=[
                            'selfsimilarity-discriminative',
                            'shared-discriminative', 'intra-discriminative'
                        ],
                        nargs='+',
                        type=str,
                        help='Decorrelations to apply between DiVA branches and respective directions (from>to).')
    parser.add_argument(
        '--diva_rho_decorrelation',
        default=[300, 300, 300],
        nargs='+',
        type=float,
        help='Weights for adversarial Separation of embeddings.')

    parser.add_argument('--diva_alpha_ssl',
                        default=0.3,
                        type=float,
                        help='Weighting for self-supervised adv. decorrelation loss.')
    parser.add_argument('--diva_alpha_shared',
                        default=0.3,
                        type=float,
                        help='Weighting for shared adv. decorrelation loss.')
    parser.add_argument('--diva_alpha_intra',
                        default=0.3,
                        type=float,
                        help='Weighting for intra adv. decorrelation loss.')

    ### Adversarial Separation Loss
    parser.add_argument('--diva_decorrnet_dim', default=512, type=int)
    parser.add_argument('--diva_decorrnet_lr', default=0.00001, type=float)

    ### (Fast) Momentum Contrast Loss
    parser.add_argument('--diva_moco_momentum', default=0.9, type=float)
    parser.add_argument('--diva_moco_temperature', default=0.01, type=float)
    parser.add_argument('--diva_moco_n_key_batches', default=30, type=int)
    parser.add_argument('--diva_moco_lower_cutoff', default=0.5, type=float)
    parser.add_argument('--diva_moco_upper_cutoff', default=1.4, type=float)
    parser.add_argument('--diva_moco_temp_lr', default=0.0005, type=float)
    parser.add_argument('--diva_moco_trainable_temp', action='store_true', help='')

    return parser


#######################################
def maxentropy_parameters(parser):
    parser.add_argument('--maxentropy_tau',
                        nargs='+',
                        default=[10000],
                        type=int)
    parser.add_argument('--maxentropy_gamma', default=0.1, type=float)
    parser.add_argument('--maxentropy_chunksize', default=128, type=int)
    parser.add_argument('--maxentropy_iter', default=10, type=int)
    parser.add_argument('--maxentropy_lrmulti', default=1, type=float)
    parser.add_argument('--maxentropy_latent', default=100, type=int)
    parser.add_argument('--maxentropy_w', default=0.1, type=float)
    parser.add_argument('--with_entropy', action='store_true')
    return parser


#######################################
def extension_parameters(parser):
    parser.add_argument(
        '--ext_svd_reg',
        default=0,
        type=float,
        help='If set, regularizes the embedding space variance')
    return parser


#######################################
def wandb_parameters(parser):
    ### Wandb Log Arguments
    parser.add_argument('--log_online', action='store_true')
    parser.add_argument('--online_backend',
                        default='wandb',
                        type=str,
                        help='Options are currently: wandb & comet')
    parser.add_argument('--wandb_key',
                        default='<your_key_here>',
                        type=str,
                        help='Options are currently: wandb & comet')
    parser.add_argument(
        '--project',
        default='Sample_Runs',
        type=str,
        help=
        'W&B project folder name.'
    )
    parser.add_argument(
        '--group',
        default='Sample_Run',
        type=str,
        help=
        'W&B group name > merges runs with multiple different seeds.'
    )

    return parser


#######################################
def loss_specific_parameters(parser):
    ### Contrastive Loss
    parser.add_argument(
        '--loss_contrastive_pos_margin',
        default=0,
        type=float,
        help='positive and negative margins for contrastive pairs.')
    parser.add_argument(
        '--loss_contrastive_neg_margin',
        default=1,
        type=float,
        help='positive and negative margins for contrastive pairs.')

    ### Triplet-based Losses
    parser.add_argument('--loss_triplet_margin',
                        default=0.2,
                        type=float,
                        help='Margin for Triplet Loss')

    ### MarginLoss
    parser.add_argument(
        '--loss_margin_margin',
        default=0.2,
        type=float,
        help='Learning Rate for class margin parameters in MarginLoss')
    parser.add_argument(
        '--loss_margin_beta_lr',
        default=0.0005,
        type=float,
        help='Learning Rate for class margin parameters in MarginLoss')
    parser.add_argument('--loss_margin_beta',
                        default=1.2,
                        type=float,
                        help='Initial Class Margin Parameter in Margin Loss')
    parser.add_argument('--loss_margin_nu',
                        default=0,
                        type=float,
                        help='Regularisation value on betas in Margin Loss.')
    parser.add_argument('--loss_margin_beta_constant', action='store_true')

    ### ProxyNCA
    parser.add_argument('--loss_proxynca_lrmulti',
                        default=50,
                        type=float,
                        help='')
    parser.add_argument('--loss_proxynca_sphereradius',
                        default=3,
                        type=float,
                        help='')
    parser.add_argument('--loss_proxynca_temperature',
                        default=1,
                        type=float,
                        help='')
    parser.add_argument('--loss_proxynca_convert_to_p',
                        action='store_true',
                        help='')
    parser.add_argument('--loss_proxynca_cosine_dist', action='store_true')
    parser.add_argument('--loss_proxynca_sq_dist', action='store_true')
    #NOTE: The number of proxies is determined by the number of data classes.

    ### ProxyAnchor
    parser.add_argument('--loss_oproxy_lrmulti',
                        default=2000,
                        type=float,
                        help='')
    parser.add_argument('--loss_oproxy_pos_alpha',
                        default=32,
                        type=float,
                        help='')
    parser.add_argument('--loss_oproxy_neg_alpha',
                        default=32,
                        type=float,
                        help='')
    parser.add_argument('--loss_oproxy_pos_delta',
                        default=0.1,
                        type=float,
                        help='')
    parser.add_argument('--loss_oproxy_neg_delta',
                        default=-0.1,
                        type=float,
                        help='')
    parser.add_argument('--loss_oproxy_mode',
                        default='anchor',
                        type=str,
                        help='')
    parser.add_argument('--loss_oproxy_euclidean',
                        action='store_true',
                        help='')
    parser.add_argument('--loss_oproxy_detach_proxies',
                        action='store_true',
                        help='')
    parser.add_argument('--loss_oproxy_warmup_it',
                        default=0,
                        type=int,
                        help='')

    ### NPair L2 Penalty
    parser.add_argument(
        '--loss_npair_l2',
        default=0.005,
        type=float,
        help=
        'L2 weight in NPair. Note: Set to 0.02 in paper, but multiplied with 0.25 in the implementation as well.'
    )

    ### Angular Loss
    parser.add_argument('--loss_angular_alpha',
                        default=45,
                        type=float,
                        help='Angular margin in degrees.')
    parser.add_argument(
        '--loss_angular_npair_ang_weight',
        default=2,
        type=float,
        help='relative weighting between angular and npair contribution.')
    parser.add_argument(
        '--loss_angular_npair_l2',
        default=0.005,
        type=float,
        help='relative weighting between angular and npair contribution.')

    ### Multisimilary Loss
    parser.add_argument('--loss_multisimilarity_pos_weight',
                        default=2,
                        type=float,
                        help='Weighting on positive similarities.')
    parser.add_argument('--loss_multisimilarity_neg_weight',
                        default=40,
                        type=float,
                        help='Weighting on negative similarities.')
    parser.add_argument(
        '--loss_multisimilarity_margin',
        default=0.1,
        type=float,
        help='Distance margin for both positive and negative similarities.')
    parser.add_argument('--loss_multisimilarity_pos_thresh',
                        default=0.5,
                        type=float,
                        help='Exponential pos. thresholding.')
    parser.add_argument('--loss_multisimilarity_neg_thresh',
                        default=0.5,
                        type=float,
                        help='Exponential neg. thresholding.')
    parser.add_argument('--loss_multisimilarity_d_mode',
                        default='cosine',
                        type=str,
                        help='Type of distances to compute.')


    ### Quadruplet Loss
    parser.add_argument('--loss_quadruplet_alpha1',
                        default=1,
                        type=float,
                        help='')
    parser.add_argument('--loss_quadruplet_alpha2',
                        default=0.5,
                        type=float,
                        help='')
    parser.add_argument('--loss_quadruplet_margin_alpha_1',
                        default=0.2,
                        type=float,
                        help='')
    parser.add_argument('--loss_quadruplet_margin_alpha_2',
                        default=0.2,
                        type=float,
                        help='')

    ### Normalized Softmax Loss
    parser.add_argument('--loss_arcface_lr',
                        default=0.0005,
                        type=float,
                        help='')
    parser.add_argument('--loss_arcface_angular_margin',
                        default=0.5,
                        type=float,
                        help='')
    parser.add_argument('--loss_arcface_feature_scale',
                        default=16,
                        type=float,
                        help='')

    return parser


#######################################
def batchmining_specific_parameters(parser):
    ### Distance-based_Sampling
    parser.add_argument('--miner_distance_lower_cutoff',
                        default=0.5,
                        type=float)
    parser.add_argument('--miner_distance_upper_cutoff',
                        default=1.4,
                        type=float)
    ### Spectrum-Regularized Miner
    parser.add_argument('--miner_rho_distance_lower_cutoff',
                        default=0.5,
                        type=float)
    parser.add_argument('--miner_rho_distance_upper_cutoff',
                        default=1.4,
                        type=float)
    parser.add_argument('--miner_rho_distance_cp', default=0.2, type=float)
    return parser


#######################################
def batch_creation_parameters(parser):
    parser.add_argument('--data_sampler',
                        default='class_random',
                        type=str,
                        help='How the batch is created.')
    parser.add_argument('--data_ssl_set', action='store_true')
    parser.add_argument(
        '--samples_per_class',
        default=2,
        type=int,
        help=
        'Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.'
    )

    return parser


#####################################
def opt_filter(opt):
    vopt = vars(opt)
    keys = list(vopt.keys())
    for key in keys:
        if 'loss_' + opt.loss not in key and 'loss_' in key:
            del vopt[key]
        if 'miner_' + opt.batch_mining not in key and 'miner_' in key:
            del vopt[key]
        if 'ext_' + opt.extension not in key and 'ext_' in key:
            del vopt[key]
