"""================= ZeroShot Generalization vs Distribution Shift. ============================="""
"""############ CUB200-2011 ###########"""
########## Split-ID 1
split_id=1
### Margin, Beta 1.2
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_Margin_b12_Distance --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_Margin_b12_Distance --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --embed_dim 512

### Multisimilarity
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_Multisimilarity --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_Multisimilarity --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize --embed_dim 512

### ArcFace
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_ArcFace --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss arcface --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_ArcFace --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss arcface --arch resnet50_frozen_normalize --embed_dim 512

### ProxyAnchor
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_oproxy --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss oproxy --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_oproxy --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss oproxy --arch resnet50_frozen_normalize --embed_dim 512

### Margin, R-D, Beta 0.6
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_RMargin_b06_Distance --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining rho_distance --miner_rho_distance_cp 0.4 --arch resnet50_frozen_normalize --loss_margin_beta 0.6 --embed_dim 512
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_RMargin_b06_Distance --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining rho_distance --miner_rho_distance_cp 0.4 --arch resnet50_frozen_normalize --loss_margin_beta 0.6 --embed_dim 512

### DiVA
python ood_diva_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_DiVA --seed 0 --gpu $gpu --bs 108 --samples_per_class 2 --loss margin --batch_mining distance --diva_rho_decorrelation 1500 1500 1500 --diva_alpha_ssl 0.3 --diva_alpha_intra 0.3 --diva_alpha_shared 0.3 --arch resnet50_frozen_normalize --embed_dim 128
python ood_diva_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_DiVA --seed 1 --gpu $gpu --bs 108 --samples_per_class 2 --loss margin --batch_mining distance --diva_rho_decorrelation 1500 1500 1500 --diva_alpha_ssl 0.3 --diva_alpha_intra 0.3 --diva_alpha_shared 0.3 --arch resnet50_frozen_normalize --embed_dim 128

### S2SD
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_S2SD --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 512  --loss_s2sd_source margin --batch_mining rho_distance --miner_rho_distance_cp 0.4 --loss_margin_beta 0.6 --loss_s2sd_target margin --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_pool_aggr --loss_s2sd_feat_distill_delay 0
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_S2SD --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 512  --loss_s2sd_source margin --batch_mining rho_distance --miner_rho_distance_cp 0.4 --loss_margin_beta 0.6 --loss_s2sd_target margin --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_pool_aggr --loss_s2sd_feat_distill_delay 0

### Uniformity
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_MaxEntropy --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --with_entropy --maxentropy_w 0.2 --embed_dim 512
python ood_main.py --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CUB_ID-1_MaxEntropy --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --with_entropy --maxentropy_w 0.2 --embed_dim 512




"""############ CARS196 ###########"""
########## Split-ID 1
split_id=1
### Margin, Beta 1.2
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_Margin_b12_Distance --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_Margin_b12_Distance --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --embed_dim 512

### Multisimilarity
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_Multisimilarity --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_Multisimilarity --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize --embed_dim 512

### ArcFace
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_ArcFace --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss arcface --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_ArcFace --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss arcface --arch resnet50_frozen_normalize --embed_dim 512

### ProxyAnchor
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_oproxy --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss oproxy --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_oproxy --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss oproxy --arch resnet50_frozen_normalize --embed_dim 512

### Margin, R-D, Beta 0.6
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_RMargin_b06_Distance --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining rho_distance --miner_rho_distance_cp 0.35 --arch resnet50_frozen_normalize --loss_margin_beta 0.6 --embed_dim 512
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_RMargin_b06_Distance --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining rho_distance --miner_rho_distance_cp 0.35 --arch resnet50_frozen_normalize --loss_margin_beta 0.6 --embed_dim 512

### DiVA
python ood_diva_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_DiVA --seed 0 --gpu $gpu --bs 108 --samples_per_class 2 --loss margin --batch_mining distance --diva_rho_decorrelation 100 100 100 --diva_alpha_ssl 0.15 --diva_alpha_intra 0.15 --diva_alpha_shared 0.15 --arch resnet50_frozen_normalize --embed_dim 128
python ood_diva_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_DiVA --seed 1 --gpu $gpu --bs 108 --samples_per_class 2 --loss margin --batch_mining distance --diva_rho_decorrelation 100 100 100 --diva_alpha_ssl 0.15 --diva_alpha_intra 0.15 --diva_alpha_shared 0.15 --arch resnet50_frozen_normalize --embed_dim 128

### S2SD
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_S2SD --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 512  --loss_s2sd_source margin --batch_mining rho_distance --miner_rho_distance_cp 0.3 --loss_margin_beta 0.6 --loss_s2sd_target margin --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_pool_aggr --loss_s2sd_feat_distill_delay 0
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_S2SD --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 512  --loss_s2sd_source margin --batch_mining rho_distance --miner_rho_distance_cp 0.3 --loss_margin_beta 0.6 --loss_s2sd_target margin --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_pool_aggr --loss_s2sd_feat_distill_delay 0

### Uniformity
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_MaxEntropy --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --with_entropy --maxentropy_w 0.2 --embed_dim 512
python ood_main.py --dataset cars196 --checkpoint --data_hardness $split_id --kernels 6 --source $datapath --n_epochs 200 --log_online --project DML_OOD-Shift_Study --group CAR_ID-1_MaxEntropy --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --with_entropy --maxentropy_w 0.2 --embed_dim 512




"""############ SOP ###########"""
########## Split-ID 1
split_id=1
### Margin, Beta 1.2
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_Margin_b12_Distance --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_Margin_b12_Distance --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --embed_dim 512

### Multisimilarity
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_Multisimilarity --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_Multisimilarity --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize --embed_dim 512

### ArcFace
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_ArcFace --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss arcface --arch resnet50_frozen_normalize --embed_dim 512
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_ArcFace --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss arcface --arch resnet50_frozen_normalize --embed_dim 512

### ProxyAnchor
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_oproxy --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss oproxy --arch resnet50_frozen_normalize --embed_dim 512 --warmup 1
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_oproxy --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss oproxy --arch resnet50_frozen_normalize --embed_dim 512 --warmup 1

### Margin, R-D, Beta 0.6
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_RMargin_b06_Distance --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining rho_distance --miner_rho_distance_cp 0.15 --arch resnet50_frozen_normalize --loss_margin_beta 0.9 --embed_dim 512
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_RMargin_b06_Distance --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining rho_distance --miner_rho_distance_cp 0.15 --arch resnet50_frozen_normalize --loss_margin_beta 0.9 --embed_dim 512

### DiVA
python ood_diva_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_DiVA --seed 0 --gpu $gpu --bs 108 --samples_per_class 2 --loss margin --batch_mining distance --diva_rho_decorrelation 150 150 150 --diva_alpha_ssl 0.2 --diva_alpha_intra 0.2 --diva_alpha_shared 0.2 --arch resnet50_frozen_normalize --embed_dim 128
python ood_diva_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_DiVA --seed 1 --gpu $gpu --bs 108 --samples_per_class 2 --loss margin --batch_mining distance --diva_rho_decorrelation 150 150 150 --diva_alpha_ssl 0.2 --diva_alpha_intra 0.2 --diva_alpha_shared 0.2 --arch resnet50_frozen_normalize --embed_dim 128

### S2SD
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_R101_ID-1_S2SD --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss_margin_beta 0.9 --loss s2sd --arch resnet101_frozen_normalize --embed_dim 512  --loss_s2sd_source margin --batch_mining distance --loss_s2sd_target margin --loss_s2sd_T 1 --loss_s2sd_w 5 --loss_s2sd_target_dims 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 5 --bs 104
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_R101_ID-1_S2SD --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss_margin_beta 0.9 --loss s2sd --arch resnet101_frozen_normalize --embed_dim 512  --loss_s2sd_source margin --batch_mining distance --loss_s2sd_target margin --loss_s2sd_T 1 --loss_s2sd_w 5 --loss_s2sd_target_dims 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 5 --bs 104

### Uniformity
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_MaxEntropy --seed 0 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --with_entropy --maxentropy_w 0.02 --maxentropy_iter 200 --maxentropy_latent 50 --embed_dim 512
python ood_main.py --dataset online_products --checkpoint --data_hardness 1 --kernels 6 --source $datapath --n_epochs 150 --log_online --project DML_OOD-Shift_Study --group SOP_ID-1_MaxEntropy --seed 1 --gpu $gpu --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --with_entropy --maxentropy_w 0.02 --maxentropy_iter 200 --maxentropy_latent 50 --embed_dim 512
