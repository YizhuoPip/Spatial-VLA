data_name=libero_10_no_noops
num_gpu=4
batch_size=4
lr=2e-4
steps=100000
grad_accumulation_steps=4
run_id=$data_name-combine
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node $num_gpu finetune.py \
--model_type vlm \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--vla_path /data1/yizhuo/Spatial-VLA/pretrained_models/configs \
--vggt_path facebook/VGGT-1B \
--data_root_dir /data1/yichi/VLA-Adapter/data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_spatial False \
--use_full_injection True \
--use_l1_regression False \
--vla_layers_align 16 \
--vggt_layers_align -1 \
--align_loss_coeff 0.5 \
--image_aug True \
--num_steps_before_decay $steps \
--max_steps $steps \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size $batch_size \
--grad_accumulation_steps $grad_accumulation_steps \
--learning_rate $lr \
--lora_rank 64 \
--wandb_entity "z-yizhuo-" \
--wandb_project "$data_name" \
--run_id_override Spatial-VLA--$data_name \
2>&1 | tee logs/train_logs/$run_id.log
# > logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 
