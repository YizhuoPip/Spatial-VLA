export PYTHONPATH=$PYTHONPATH:/data1/yizhuo/Spatial-VLA/LIBERO


CUDA_VISIBLE_DEVICES=2 python /data1/yizhuo/Spatial-VLA/eval/libero/run_libero_eval.py \
--model_type vla \
--use_proprio True \
--num_images_in_input 2 \
--use_film False \
--pretrained_checkpoint /data1/yizhuo/Spatial-VLA/outputs/Spatial-VLA--libero_10_no_noops--100000_chkpt \
--task_suite_name libero_10 \
--use_l1_regression True \
2>&1 | tee logs/test_logs/Long--chkpt-100000.log