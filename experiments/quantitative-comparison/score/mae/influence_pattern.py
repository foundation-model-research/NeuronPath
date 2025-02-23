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
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

import modeling_mae_ip
import modeling_mae

import torch.nn.functional as F

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def scaled_input(emb, ig_total_step):
    baseline = torch.zeros_like(emb)  # (b, ffn_size)

    num_points = ig_total_step

    step = (emb - baseline) / num_points  # (b, ffn_size)
    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points * bs, ffn_size)
    return res, step

def path_scaled_input(ffn_path, path, ig_total_step):
    steps = []
    ress = []
    for l, emb in enumerate(ffn_path):
        p = path[l]
        baseline = emb.clone()
        for i, b in enumerate(p):
            baseline[i, :, int(b)] = 0  # (b, ffn_size)

        num_points = ig_total_step

        step = (emb - baseline) / num_points  # (b, ffn_size)
        res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points * bs, ffn_size)
        
        steps.append(step)
        ress.append(res)
    return ress, steps

def j_a_score(args, path, grad, step):
    ig_pred = []
    for g in grad:
        g = g.reshape(args.ig_total_step, args.batch_size, *g.shape[-2:]) 
        ig_pred.append(g)
        
    ja = None
    s = None

    for l, p in enumerate(path):
        temp = torch.zeros_like(ig_pred[l][:, :, :, [0]])
        for i, b in enumerate(p):
            temp[:, i] = ig_pred[l][:, i, :, [b]]
        ja = temp if ja is None else ja + temp
        
    ja = ja.sum(dim=0)

    for l, p in enumerate(path):
        temp_s = torch.zeros_like(step[l][:, :, [0]])
        for i, b in enumerate(p):
            temp_s[i] = step[l][i, :,[b]]
        s = temp_s if s is None else s + temp_s
    ja = ja * s
    
    # sum for n_tokens
    ja = ja.sum(dim=1)

    return ja

def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = ig_list
    max_ig = ig.max()

    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


def main():
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
                        default=1,
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

    # parse arguments
    args = parser.parse_args()

    # set device
    device = torch.device("cuda:0")
    n_gpu = torch.cuda.device_count()
    
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # save args
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, args.output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    
    model = modeling_mae_ip.__dict__['vit_base_patch16'](
        num_classes=1000,
        global_pool=True,
    )
    
    checkpoint = torch.load(args.pretrain_weight, map_location='cuda')
    
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # _model 
    _model = modeling_mae.__dict__['vit_base_patch16'](
        num_classes=1000,
        global_pool=True,
    )
    
    _model.load_state_dict(checkpoint['model'])
    _model.to(device)
    
    # data parallel
    if n_gpu > 1:
        _model = torch.nn.DataParallel(_model)
    _model.eval()

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
        testset = datasets.ImageFolder(root="./imagenet2012/val",
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

    for step, batch in enumerate(epoch_iterator):
        
        tic = time.perf_counter()
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        l = y[0]

        with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '-' + str(int(l)) + '.rlt' + '.jsonl'), 'a') as fw:
            res_dict = {
                'path':[],
                'score':[]
            }
            path = np.zeros((args.batch_size, model.depth)) - 1
            
            input_seq, _ = scaled_input(x, args.ig_total_step)
            
            _, ffn_weights = model(input_seq, tgt_layer=-2) 

            score = 1
            for tgt_layer in range(model.depth):
                pred_label = int(l)
                ig_pred = None

                _, grad = model(input_seq, tgt_layer=tgt_layer, tgt_label=pred_label, path=path, path_weights=ffn_weights[:(tgt_layer + 1)])  # (step * batch, ffn_size)
                new_p = torch.argmax(grad.mean(dim=0).mean(dim=0))
                path[:, tgt_layer] = new_p.detach().cpu().numpy()
                
                score *= grad.mean(dim=0).mean(dim=0)[new_p]
            step_size = args.ig_total_step
            x = x.repeat(step_size, 1, 1, 1)
            _, ffn_path = model(x, tgt_layer=-2)

            ffn_path = [_[:args.batch_size] for _ in ffn_path]

            path_scaled_weights, path_weights_steps = path_scaled_input(ffn_path, np.transpose(path, (1, 0)).tolist(), args.ig_total_step)  
            for w in path_scaled_weights:
                w.requires_grad_(True)
            torch.cuda.synchronize()
            _, grad = _model(x, tgt_layer=model.depth-1, tmp_score=path_scaled_weights, tgt_label=int(pred_label))  # (step * batch, ffn_size)

            ja_score = j_a_score(args, np.transpose(path, (1, 0)).tolist(), grad, path_weights_steps)

            res_dict['path'] = path.tolist()
            # need to /196.
            res_dict['score'] = (ja_score/196.).tolist()

            fw.write(res_dict)

            toc = time.perf_counter()
            print(f"***** Relation: {str(int(l))} evaluated. Costing time: {toc - tic:0.4f} seconds *****")


if __name__ == "__main__":
    main()  