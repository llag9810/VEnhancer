N_PARALLEL_GPUS=4

torchrun --nproc_per_node=${N_PARALLEL_GPUS} enhance_a_video_MultiGPU.py \
--up_scale 4 --target_fps 24 --noise_aug 250 \
--solver_mode 'fast' --steps 15 \
--input_path prompts/astronaut.mp4 \
--prompt 'An astronaut flying in space, featuring a steady and smooth perspective' \
--save_dir "results_with_${N_PARALLEL_GPUS}gpu/"
