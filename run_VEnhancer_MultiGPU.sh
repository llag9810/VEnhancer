N_PARALLEL_GPUS=8

torchrun --nproc_per_node=${N_PARALLEL_GPUS} enhance_a_video_MultiGPU.py \
--version v2 \
--target_fps 24 --noise_aug 250 \
--solver_mode 'normal' --steps 10 \
--input_path input \
--prompt_path input/text_prompts.txt \
--save_dir "results_with_${N_PARALLEL_GPUS}gpu/"
