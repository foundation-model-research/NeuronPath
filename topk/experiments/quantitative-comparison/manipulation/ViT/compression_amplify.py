import itertools
import logging
import argparse
import math
import os
from pprint import pprint
import re
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time
import sys

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Subset
import torch.nn.functional as F

from tqdm import tqdm

from modeling_VIT_edit import VisionTransformer, CONFIGS

import torch.nn.functional as F
from tqdm import tqdm

from torch import nn
from scipy.stats import mode


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir",
                        default='../results/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_prefix",
                        default='imgnet_test',
                        type=str,
                        help="The output prefix to indentify each running of experiment. ")

    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # parameters about integrated grad
    parser.add_argument("--ig_total_step",
                        default=20,
                        type=int,
                        help="Total step for cut.")
    parser.add_argument("--batch_size",
                        default=10,
                        type=int,
                        help="batch size")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32"],
                        default="ViT-L_16",
                        help="Which model to use.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet1k"], default="imagenet1k")
    parser.add_argument("--dataset_path", default="../dataset/imagenet2012/val")
    parser.add_argument("--pretrain_weight", type=str, default='../ckpt/ViT-B_16-224.npz',
                        help="path to the pretrain weight")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")         
    parser.add_argument("--method",
                        type=str,
                        default='ja',
                        help="Method Type, choose from ja, influence_pattern and base")
    parser.add_argument("--ja_path",
                        type=str,
                        help="Output directory for Activation and Ours method")
    parser.add_argument("--influence_pattern_path",
                        type=str,
                        help="Output directory for Influence Pattern method")   
    
    parser.add_argument("--depth_prune",
                        type=int,
                        help="Only prune for the first x layers")   
    parser.add_argument("--edit_operation",
                        type=str,
                        help='[remove, enhance, prune]')   
    
    
    
    # parse arguments
    args = parser.parse_args()
    
    match = re.search(r'p(\d+)$', args.output_dir) 
    if match:
        mask_percentage = int(match.group(1))  
        args.maskProsion = mask_percentage / 100.0  
        print(f"maskProsion is {args.maskProsion}")

    match = re.search(r'top(\d+)(?!.*top)', args.output_dir)
    if match:
        args.topk = int(match.group(1))
    print("="*20)
    print(args)
    return args


@torch.no_grad()
def main():
    args = parse_args()

    # set device
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # save args
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, args.output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = VisionTransformer(CONFIGS[args.model_type], args.img_size, zero_head=False, num_classes=1000)
    
    model.load_from(np.load(args.pretrain_weight))

    model = nn.DataParallel(model)
    model.to(device)

    model.eval()

    JA_PATH = args.ja_path
    INFLUENCE_PATTERN_PATH = args.influence_pattern_path

    methods_dict = {
        "ja": JA_PATH,
        "influence_pattern": INFLUENCE_PATTERN_PATH,
        "base": JA_PATH,
    }

    transform_cifar_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_cifar_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_imgnet = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.dataset == "cifar10":
        testset = datasets.CIFAR10(root=args.dataset_path,
                                    train=False,
                                    download=True,
                                    transform=transform_cifar_test)

    elif args.dataset == "cifar100":
        testset = datasets.CIFAR100(root=args.dataset_path,
                                    train=False,
                                    download=True,
                                    transform=transform_cifar_test)
    elif args.dataset == "imagenet1k":
        testset = datasets.ImageFolder(root=args.dataset_path,
                                            transform=transform_imgnet)
        
    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=args.batch_size,
                            num_workers=2,
                            pin_memory=True) if testset is not None else None

    epoch_iterator = tqdm(test_loader,
                        desc="Imgnet",
                        bar_format="{l_bar} {r_bar}",
                        dynamic_ncols=True,
                        )
    skipCounter=0
    def get_latest_pred_label_index(output_dir, output_prefix):
        pattern = re.compile(rf'{re.escape(output_prefix)}-(\d+)\.rlt\.jsonl')

        max_index = 0

        for file_name in os.listdir(output_dir):
            match = pattern.match(file_name)
            if match:
                pred_label = int(match.group(1))
                if pred_label > max_index:
                    max_index = pred_label

        return max_index
    skipCounter = get_latest_pred_label_index(args.output_dir, args.output_prefix)
    print(f"should skip {skipCounter} times")
    
    for batch_idx, batch in enumerate(itertools.islice(epoch_iterator, skipCounter, None)):
        tic = time.perf_counter()
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        l = y[0]
        res_dict_bag = []

        inclass_idx = batch_idx*args.batch_size - 50*int(l)
        
        
        with open(os.path.join(args.ja_path, f'imgnet_all-{int(l)}.rlt.jsonl' ), 'r') as f:
            reader = jsonlines.Reader(f)
            one_data = list(reader)
        
            
            path = []
                        
            for one_one_data in one_data:
                if args.method=="base":
                    path.append(one_one_data['base'])
                else:
                    path.append(one_one_data['ja'])

        # 50 examples summarize to 1
        path=np.array(path)
        path=path[:40,:args.depth_prune,:args.topk] 
        def most_frequent(arr):
            mode_values, _ = mode(arr, axis=0)
            return mode_values
        
        compressed_array = np.apply_along_axis(most_frequent, axis=0, arr=path) 
        broadcasted_array = np.tile(compressed_array, (10, 1, 1))
        
        
        path = torch.tensor(broadcasted_array).to(device)
        path=path[:,:,:args.topk] 
        
        
        logits, *_ = model(x[-11:-1,:,:,:], path = path, edit_operation = args.edit_operation,maskProsion=args.maskProsion)
        prune_logits = F.softmax(logits, dim=1)[:, l]

        base_logits=prune_logits
        

        with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '-' + str(int(l)) + '.rlt' + '.jsonl'), 'a') as fw:
            for _prune , _base, p in zip(prune_logits, base_logits, path):
                res_dict = {
                    'ja': [],
                    'remove': [],
                    'base': [],
                    'enhance': [],
                    "prune": []
                }
                
                res_dict['base'].append(float(_base))
                res_dict['prune'].append(float(_prune))
                
                res_dict['ja'].append(p.tolist())
                res_dict_bag.append(res_dict)
                
            fw.write(res_dict_bag)



if __name__ == "__main__":
    main()