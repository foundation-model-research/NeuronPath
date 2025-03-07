CUDA_VISIBLE_DEVICES=0,1,2,3 python ./compression_amplify_accuracy.py \
    --output_prefix imgnet_all \
    --output_dir ./edit_ip_ViT_B_32_accuracy/ \
    --model_type ViT-B_32 \
    --batch_size 10 \
    --pretrain_weight ./ckpt/ViT-B_32.npz \
    --gpus 4 \
    --dataset_path your-dataset-path \
    --ja_path path-from-step-4 \
    --influence_pattern_path path-from-step-4 \
    --method influence_pattern \