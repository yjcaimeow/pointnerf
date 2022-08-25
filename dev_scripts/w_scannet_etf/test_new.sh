#!/bin/bash
nrCheckpoint="../checkpoints"
nrDataRoot="/mnt/cache/caiyingjie/data/scannet"
name=$2
resume_iter=latest

data_root="${nrDataRoot}/scans/"
scan="scene0006_00"
scans="scene0101_00 scene0006_00 "
#scans="scene0003_00 scene0005_00 scene0006_00 scene0101_00 scene0032_00 scene0034_00 scene0035_00 scene0036_00 scene0039_00"
#scans=$3

load_points=2
feat_grad=1
conf_grad=1
dir_grad=1
color_grad=1
vox_res=900
normview=0
prune_thresh=-1
prune_iter=-1

feedforward=0
ref_vid=0
bgmodel="no" #"plane"
depth_occ=0
depth_vid="0"
trgt_id=0
manual_depth_view=1
init_view_num=3
pre_d_est="${nrCheckpoint}/MVSNet/model_000014.ckpt"
manual_std_depth=0.0
depth_conf_thresh=0.8
geo_cnsst_num=0
edge_filter=10 # pixels crop out at image edge

appr_feature_str0="imgfeat_0_0123 dir_0 point_conf"
point_conf_mode="1" # 0 for only at features, 1 for multi at weight
point_dir_mode="1" # 0 for only at features, 1 for color branch
point_color_mode="1" # 0 for only at features, 1 for color branch
default_conf=-1

agg_feat_xyz_mode="None"
agg_alpha_xyz_mode="None"
agg_color_xyz_mode="None"
feature_init_method="rand" #"rand" # "zeros"
agg_axis_weight=" 1. 1. 1."
agg_dist_pers=20
radius_limit_scale=4
depth_limit_scale=0
vscale=" 2 2 2 "
kernel_size=" 3 3 3 "
query_size=" 3 3 3 "
vsize=" 0.064 0.064 0.064 " #" 0.005 0.005 0.005 "
wcoord_query=1
z_depth_dim=400
max_o=610000
ranges=" -10.0 -10.0 -10.0 10.0 10.0 10.0 "
SR=24
K=8
P=26
NN=2

act_type="LeakyReLU"

agg_intrp_order=2
agg_distance_kernel="linear" #"avg" #"feat_intrp"
weight_xyz_freq=2
weight_feat_dim=8

point_features_dim=32
shpnt_jitter="passfunc" #"uniform" # uniform gaussian

which_agg_model="viewmlp"
apply_pnt_mask=1
shading_feature_mlp_layer0=1 #2
shading_feature_mlp_layer1=2 #2
shading_feature_mlp_layer2=0 #1
shading_feature_mlp_layer3=2 #1
shading_alpha_mlp_layer=1
shading_color_mlp_layer=4
shading_feature_num=256
dist_xyz_freq=5
num_feat_freqs=3
dist_xyz_deno=0

raydist_mode_unit=1
dataset_name='scannet_ft'
pin_data_in_memory=1
model='mvs_points_volumetric'
near_plane=0.1
far_plane=8.0
which_ray_generation='near_far_linear' #'nerf_near_far_linear' #
domain_size='1'
dir_norm=0

which_tonemap_func="off" #"gamma" #
which_render_func='radiance'
which_blend_func='alpha'
out_channels=4

num_pos_freqs=10
num_viewdir_freqs=4 #6

random_sample='random'
random_sample_size=56 # 32 * 32 = 1024

batch_size=1

plr=0.02
lr=0.0005 # 0.0005 #0.00015
lr_policy="iter_exponential_decay"
lr_decay_iters=1000000
lr_decay_exp=0.1

checkpoints_dir="${nrCheckpoint}/scannet/"
resume_dir="${nrCheckpoint}/init/dtu_dgt_d012_img0123_conf_agg2_32_dirclr20"

save_point_freq=10000 #301840 #1
maximum_step=800000 #500000 #250000 #800000

niter=10000 #1000000
niter_decay=10000 #250000

train_and_test=0 #1
test_num=10
print_freq=40
test_num_step=1

prob_maximum_step=150002 #500000 #250000 #800000
prob_freq=100
prob_num_step=2
prob_kernel_size=" 3 3 3 1 1 1 "

maximum_epoch=2000 #500000 #250000 #800000
prob_tiers="100 200 300 2000"
#prob_tiers="30 90 150 210 270 2000"
test_freq=50 #  #100 #1200 #1200 #30184 #30184 #50000
save_iter_freq=50

prob_mode=1 # 0, n, 1 t, 10 t&n
prob_thresh=0.9
prob_mul=0.4
zero_epsilon=1e-3

visual_items='coarse_raycolor gt_image sample_loc sample_loc_w ray_valid decoded_features'
zero_one_loss_items='conf_coefficient' #regularize background to be either 0 or 1
zero_one_loss_weights=" 0.0001 "
sparse_loss_weight=0

color_loss_weights="0.1 0.1 1.0 "
color_loss_items='loss_rgb loss_alpha ray_masked_coarse_raycolor '
test_color_loss_items='coarse_raycolor ray_miss_coarse_raycolor ray_masked_coarse_raycolor'

bg_color="white" #"0.0,0.0,0.0,1.0,1.0,1.0"
split="train"

n_threads=20
PART=pat_taurus
GPUNUM=1
PROCESSNUM=1
embed_init_type='model'

progressive_distill=1
agg_type='attention'
k_type='knn'
ddp_init_type='new'

port=$1

cd run

TOOLS="srun --partition=$PART --quotatype=auto --preempt -n${PROCESSNUM} --gres=gpu:${GPUNUM} --ntasks-per-node=${GPUNUM} --cpus-per-task=5"
$TOOLS --job-name=$JOBNAME sh -c "python -m torch.distributed.launch train.py \
        --progressive_distill ${progressive_distill} \
        --only_render \
        --embed_init_type ${embed_init_type} \
        --ddp_init_type ${ddp_init_type} \
        --k_type ${k_type} \
        --ddp_train --port ${port} \
        --name $name \
        --scan $scan \
        --scans $scans \
        --agg_type ${agg_type} \
        --data_root $data_root \
        --dataset_name $dataset_name \
        --model $model \
        --which_render_func $which_render_func \
        --which_blend_func $which_blend_func \
        --out_channels $out_channels \
        --num_pos_freqs $num_pos_freqs \
        --num_viewdir_freqs $num_viewdir_freqs \
        --random_sample $random_sample \
        --random_sample_size $random_sample_size \
        --batch_size $batch_size \
        --maximum_epoch $maximum_epoch \
        --maximum_step $maximum_step \
        --prob_maximum_step $prob_maximum_step \
        --plr $plr \
        --lr $lr \
        --lr_policy $lr_policy \
        --lr_decay_iters $lr_decay_iters \
        --lr_decay_exp $lr_decay_exp \
        --checkpoints_dir $checkpoints_dir \
        --save_iter_freq $save_iter_freq \
        --niter $niter \
        --niter_decay $niter_decay \
        --n_threads $n_threads \
        --pin_data_in_memory $pin_data_in_memory \
        --train_and_test $train_and_test \
        --test_num $test_num \
        --test_freq $test_freq \
        --test_num_step $test_num_step \
        --test_color_loss_items $test_color_loss_items \
        --prob_freq $prob_freq \
        --prob_num_step $prob_num_step \
        --print_freq $print_freq \
        --bg_color $bg_color \
        --split $split \
        --which_ray_generation $which_ray_generation \
        --near_plane $near_plane \
        --far_plane $far_plane \
        --dir_norm $dir_norm \
        --which_tonemap_func $which_tonemap_func \
        --load_points $load_points \
        --resume_dir $resume_dir \
        --resume_iter $resume_iter \
        --feature_init_method $feature_init_method \
        --agg_axis_weight $agg_axis_weight \
        --agg_distance_kernel $agg_distance_kernel \
        --radius_limit_scale $radius_limit_scale \
        --depth_limit_scale $depth_limit_scale  \
        --vscale $vscale    \
        --kernel_size $kernel_size  \
        --SR $SR  \
        --K $K  \
        --P $P \
        --NN $NN \
        --agg_feat_xyz_mode $agg_feat_xyz_mode \
        --agg_alpha_xyz_mode $agg_alpha_xyz_mode \
        --agg_color_xyz_mode $agg_color_xyz_mode  \
        --save_point_freq $save_point_freq  \
        --raydist_mode_unit $raydist_mode_unit  \
        --agg_dist_pers $agg_dist_pers \
        --agg_intrp_order $agg_intrp_order \
        --shading_feature_mlp_layer0 $shading_feature_mlp_layer0 \
        --shading_feature_mlp_layer1 $shading_feature_mlp_layer1 \
        --shading_feature_mlp_layer2 $shading_feature_mlp_layer2 \
        --shading_feature_mlp_layer3 $shading_feature_mlp_layer3 \
        --shading_feature_num $shading_feature_num \
        --dist_xyz_freq $dist_xyz_freq \
        --shpnt_jitter $shpnt_jitter \
        --shading_alpha_mlp_layer $shading_alpha_mlp_layer \
        --shading_color_mlp_layer $shading_color_mlp_layer \
        --which_agg_model $which_agg_model \
        --color_loss_weights $color_loss_weights \
        --num_feat_freqs $num_feat_freqs \
        --dist_xyz_deno $dist_xyz_deno \
        --apply_pnt_mask $apply_pnt_mask \
        --point_features_dim $point_features_dim \
        --color_loss_items $color_loss_items \
        --feedforward $feedforward \
        --trgt_id $trgt_id \
        --depth_vid $depth_vid \
        --ref_vid $ref_vid \
        --manual_depth_view $manual_depth_view \
        --pre_d_est $pre_d_est \
        --depth_occ $depth_occ \
        --manual_std_depth $manual_std_depth \
        --visual_items $visual_items \
        --appr_feature_str0 $appr_feature_str0 \
        --init_view_num $init_view_num \
        --feat_grad $feat_grad \
        --conf_grad $conf_grad \
        --dir_grad $dir_grad \
        --color_grad $color_grad \
        --depth_conf_thresh $depth_conf_thresh \
        --bgmodel $bgmodel \
        --vox_res $vox_res \
        --act_type $act_type \
        --geo_cnsst_num $geo_cnsst_num \
        --point_conf_mode $point_conf_mode \
        --point_dir_mode $point_dir_mode \
        --point_color_mode $point_color_mode \
        --normview $normview \
        --prune_thresh $prune_thresh \
        --prune_iter $prune_iter \
        --sparse_loss_weight $sparse_loss_weight \
        --zero_one_loss_items $zero_one_loss_items \
        --zero_one_loss_weights $zero_one_loss_weights \
        --default_conf $default_conf \
        --edge_filter $edge_filter \
        --vsize $vsize \
        --wcoord_query $wcoord_query \
        --ranges $ranges \
        --z_depth_dim $z_depth_dim \
        --max_o $max_o \
        --prob_thresh $prob_thresh \
        --prob_mul $prob_mul \
        --prob_kernel_size $prob_kernel_size \
        --prob_tiers $prob_tiers \
        --query_size $query_size \
        --debug"
