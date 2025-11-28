data_name=libero_spatial_no_noops
num_gpu=2
batch_size=4
lr=2e-4
steps=150000
grad_accumulation_steps=8
run_id=$data_name-combine
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nnodes 1 --nproc-per-node $num_gpu finetune.py \
--model_type vla \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--vla_path openvla/openvla-7b \
--data_root_dir /data-root \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_spatial True \
--vla_layers_align 24 \
--vggt_layers_align -1 \
--align_loss_coeff 0.5 \
--image_aug True \
--num_steps_before_decay $steps \
--max_steps $steps \
--save_freq 10000 \
--save_latest_checkpoint_only True \
--merge_lora_during_training False \
--batch_size $batch_size \
--grad_accumulation_steps $grad_accumulation_steps \
--learning_rate $lr \
--lora_rank 32 \
--wandb_entity "z-yizhuo-" \
--wandb_project "$data_name" \
--run_id_override VLA-Adapter--$data_name\
2>&1 | tee logs/train_logs/$run_id.log
# > logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 
