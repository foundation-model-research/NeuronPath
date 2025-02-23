CUDA_VISIBLE_DEVICES=0 python influence_pattern.py \
    --output_prefix imgnet_all \
    --output_dir ./MAE-B-ip/ \
    --ig_total_step 20 \
    --pretrain_weight your-mae-path \
