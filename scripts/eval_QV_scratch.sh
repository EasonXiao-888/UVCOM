export CUDA_VISIBLE_DEVICES=7
export CUDA_LAUNCH_BLOCKING=1

dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=path_to_save/qvhighlights_results
device=1
enc_layers=3
dec_layers=3
query_num=30
n_txt_mu=5
n_visual_mu=30

span_loss_type=l1
sim_loss_coef=1
neg_loss_coef=0.5
exp_id=test
seed=2023
lr=1e-4
lr_gamma=0.1
neg_choose_epoch=80
lr_drop=100
resume=path_to_ckpt/QV_scratch.ckpt

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=path_to_qv_features/features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32


PYTHONPATH=$PYTHONPATH:. python uvcom/eval.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--num_queries ${query_num} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--seed ${seed} \
--lr_gamma ${lr_gamma} \
--dec_layers ${dec_layers} \
--lr_drop ${lr_drop} \
--em_iter 5 \
--n_txt_mu ${n_txt_mu} \
--n_visual_mu ${n_visual_mu} \
--resume ${resume} \
--neg_choose_epoch ${neg_choose_epoch}\
${@:1}
