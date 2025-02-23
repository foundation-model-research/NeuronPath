CUDA_VISIBLE_DEVICES=0,1,2,3 python ./ip_analysis_parallel.py \
    --output_prefix imgnet_all \
    --output_dir ./ip_ja_ViT_L_32/ \
    --model_type ViT-L_32 \
    --batch_size 5 \
    --pretrain_weight PATH_TO_CKPT \
    --gpus 4 \
    --method influence_pattern \
