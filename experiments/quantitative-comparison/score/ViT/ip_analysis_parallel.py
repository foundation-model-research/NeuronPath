import logging
import argparse
import math
import os
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
import torch.nn as nn

from tqdm import tqdm

from modeling_VIT import VisionTransformer, CONFIGS

INFLUENCE_PATTERN_PATH = 'PATH_TO_INFLUENCE_PATTERN_RESULTS'

def path_scaled_input(ffn_path, path, ig_total_step):
    # emb: (b, ffn_size)
    # print("emb:{}".format(emb.shape))
    steps = []
    ress = []
    for l, emb in enumerate(ffn_path):
        p = path[l]
        baseline = emb.clone()
        for i, b in enumerate(p):
            baseline[i, b] = 0  # (b, ffn_size)
        # print("baseline:{}".format(baseline.shape))
        num_points = ig_total_step
        
        step = (emb - baseline) / num_points  # (b, ffn_size)
        # print(step[i, b])
        # print("step:{}".format(step.shape))
        res = torch.cat([torch.add(baseline, step * i).unsqueeze(0) for i in range(num_points)], dim=0)  # (num_points * bs, ffn_size)

        steps.append(step)
        ress.append(res)
    return ress, steps

def j_a_score(args, path, grad, step):
    ig_pred = []
    for g in grad:
        g = g.reshape(args.ig_total_step, args.batch_size, -1) 
        ig_pred.append(g)
        
    ja = None
    s = None
    for l, p in enumerate(path):
        temp = torch.zeros_like(ig_pred[l][:,:,[0]])
        for i, b in enumerate(p):
            temp[:, i] = ig_pred[l][:, i, [b]]
        ja = temp if ja is None else ja + temp
        
    ja = ja.sum(dim=0)

    for l, p in enumerate(path):
        temp_s = torch.zeros_like(step[l][:,[0]])
        for i, b in enumerate(p):
            temp_s[i] = step[l][i, [b]]
        s = temp_s if s is None else s + temp_s
    ja = ja * s

    return ja

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
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-L_16",
                        help="Which model to use.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet1k"], default="imagenet1k",
                        help="Which downstream task.")
    parser.add_argument("--pretrain_weight", type=str, default='',
                        help="path to the pretrain weight")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")         
    parser.add_argument("--method",
                        type=str,
                        default='ja',
                        help="Method Type, choose from influence_pattern and ''.")    

    # parse arguments
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # set device
    device = torch.device("cuda:0")
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
        testset = datasets.CIFAR10(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_cifar_test)

    elif args.dataset == "cifar100":
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=False,
                                    transform=transform_cifar_test)
    elif args.dataset == "imagenet1k":
        testset = datasets.ImageFolder(root="./data/imagenet2012/val",
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

    for batch_idx, batch in enumerate(epoch_iterator):
        tic = time.perf_counter()
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        l = y[0]
        res_dict_bag = []

        inclass_idx = batch_idx*args.batch_size - 50*int(l)
        
        if args.method == "influence_pattern":
            with open(os.path.join(INFLUENCE_PATTERN_PATH, f"imgnet_all-{int(l)}.rlt.jsonl"), 'r') as f:
                reader = jsonlines.Reader(f)
                _data = list(reader)
                
                data = _data
                sub_data = data[inclass_idx:inclass_idx+args.batch_size]
                
                path = []
                ffn_weights = []

                for one_data in sub_data:
                    path.append([int(n) for n in one_data['path'][0]])
                    # ffn_weights.append(ffn_weight)

                for layer_id in range(model.module.config.transformer.num_layers):
                    torch.cuda.synchronize()
                    *_, ffn_weight = model(x, tgt_layer = layer_id)

                    ffn_weights.append(ffn_weight.detach())
                ffn_weights = torch.cat([t.unsqueeze(1) for t in ffn_weights], dim=1)


        elif args.method == "":
            pass

        with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '-' + str(int(l)) + '.rlt' + '.jsonl'), 'a') as fw:
            path_scaled_weights, path_weights_steps = path_scaled_input(ffn_weights, path, args.ig_total_step)  
            
            d_chunk = len(path_scaled_weights)
            path_scaled_weights = torch.cat([t for t in path_scaled_weights], dim = 0)
            # for w in path_scaled_weights:
                # w.requires_grad_(True)

            path_scaled_weights.requires_grad = True

            # path_scaled_weights = torch.cat([t for t in path_scaled_weights], dim=0)
            # path = torch.tensor(path).to(device)

            grads = []
            
            # for input_x, input_w in zip(x, path_scaled_weights):
                # input_w.requires_grad_(True)
                # torch.cuda.synchronize()
                # _, grad = model(input_x.unsqueeze(0).expand(n_gpu, -1, -1, -1), tgt_layer=model.module.config.transformer.num_layers-1, tmp_score=input_w, tgt_label=int(l))  # (step * batch, ffn_size)
            step_size = path_scaled_weights.shape[0] // x.shape[0]
            x = torch.repeat_interleave(x, step_size, dim=0) # x.repeat(step_size, 1, 1, 1)
        

            

            torch.cuda.synchronize()
            _, grad = model(x, tgt_layer=model.module.config.transformer.num_layers-1, tmp_score=path_scaled_weights, tgt_label=int(l))  # (step * batch, ffn_size)
            
            grads.append(grad[0].reshape(d_chunk, 
                                         -1, 
                                         grad[0].shape[-2], 
                                         grad[0].shape[-1]))
            
            ja_scores = []

            _path_weight_step = 0.0

            with torch.no_grad():
                # change the path method
                for _path, _grad, _path_weight_step in zip(path, grads[0], path_weights_steps):
                    _w_sum = 0.0
                    for layer_id, neuron_idx in enumerate(_path):
                        _w_sum += _path_weight_step[layer_id, neuron_idx]
                    for layer_id, neuron_idx in enumerate(_path):
                        _path_weight_step[layer_id, neuron_idx] = _w_sum

                    ja_scores.append(float((_grad.sum(dim=0) * _path_weight_step).sum().cpu().numpy()))
                # score = j_a_score(args, path, grads, path_weights_steps)

            for p_s, p in zip(ja_scores, path):
                res_dict = {
                    'ja': [],
                    'ja_score': [],
                }
                
                res_dict['ja_score'].append(p_s)
                res_dict['ja'].append(p)
                res_dict_bag.append(res_dict)


            fw.write(res_dict_bag)
            # record running time
            toc = time.perf_counter()
            print(f"***** Relation: {str(int(l))} evaluated. Costing time: {toc - tic:0.4f} seconds *****")
            # sys.exit()

if __name__ == "__main__":
    main()
        
