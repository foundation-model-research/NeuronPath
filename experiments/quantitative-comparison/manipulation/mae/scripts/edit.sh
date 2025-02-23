CUDA_VISIBLE_DEVICES=0,1,2,3 python ./compression_amplify.py \
    --output_prefix imgnet_all \
    --output_dir ./edit-MAE-B-ip/ \
    --batch_size 50 \
    --pretrain_weight ./ckpt/mae_finetuned_vit_base.pth \
    --gpus 4 \
    --dataset_path your-dataset-path \
    --ja_path path-from-step-5 \
    --influence_pattern_path path-from-step-5 \
    --method influence_pattern \