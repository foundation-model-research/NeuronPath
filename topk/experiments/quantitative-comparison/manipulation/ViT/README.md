# Experiments on ViT

## Run Accuracy Deviation
```
bash ./scripts/edit_accuracy.sh
```

`model_type`: Choose one from ['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32'].

`pretrain_weight`: Checkpoint path of ViT model.

`dataset_path`: Your Imagenet-1k val dataset path.

`ja_path`: Joint Attribution Score `output_dir` of Step 4.

`ip_path`: Influence Pattern, `output_dir` of Step 4.

`method`: Choose one from ['ja', 'influence_pattern', 'base'].

## Run Probability Deviation or Prune 
```
bash ./scripts/edit_logits.sh
```

`output_dir`: Output path of your experiment. You need to write the path as '\*top{α}-p{‌β}\*'. Where α is an integer chosen from 1-5. ‌β is a float chosen from 1-10, e.g. '\*top1-p1\*'

`model_type`: Choose one from ['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32'].

`pretrain_weight`: Checkpoint path of ViT model.

`dataset_path`: Your Imagenet-1k val dataset path.

`ja_path`: Joint Attribution Score `output_dir` of Step 4.

`ip_path`: Influence Pattern, `output_dir` of Step 4.

`method`: Choose one from ['ja', 'influence_pattern', 'base'].

`edit_operation`: Choose one from ['remove', 'enhance', 'prune']


