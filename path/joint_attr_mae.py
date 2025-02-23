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

import sys
sys.path.append("../experiments/quantitative-comparison/score/mae")
import modeling_mae

import torch.nn.functional as F
import torch.utils.checkpoint as cp

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def scaled_input(emb, ig_total_step):
    baseline = torch.zeros_like(emb)  # (b, n_tokens, ffn_size)
    num_points = ig_total_step

    step = (emb - baseline) / num_points  # (b, n_tokens, ffn_size)
    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points * bs, ffn_size)
    return res, step

def path_scaled_input(ffn_path, path, ig_total_step):
    steps = []
    ress = []
    for l, emb in enumerate(ffn_path):
        p = path[l]
        baseline = emb.clone()
        for i, b in enumerate(p):
            baseline[i, :, b] = 0  # (b, ffn_size)

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
    # ig = np.array(ig_list)  # (n_layer, ffn_size)
    ig = ig_list
    # print("ig:{}".format(ig.shape))
    max_ig = ig.max()

    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            '''
                TODO: threshold!!!
            '''
            # if ig[i][j] >= max_ig * 0.1:
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
    parser.add_argument("--batch_size", choices=[1, 2, 5, 10, 25, 50],
                        default=10,
                        type=int,
                        help="batch size")
    parser.add_argument("--model_type", 
                        default="",
                        help="Which model to use.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet1k"], default="imagenet1k")
    parser.add_argument("--dataset_path", default="../dataset/imagenet2012/val")
    parser.add_argument("--pretrain_weight", type=str, default='../ckpt/mae_finetuned_vit_base.pth',
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
    
    model = modeling_mae.__dict__['vit_base_patch16'](
        num_classes=1000,
        global_pool=True,
    )

    checkpoint = torch.load(args.pretrain_weight, map_location='cuda')

    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # data parallel
    model = torch.nn.DataParallel(model)
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

    for batch_idx, batch in enumerate(epoch_iterator):
        tic = time.perf_counter()
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        pred_label = y[0]
        res_dict_bag = []

        with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '-' + str(int(pred_label)) + '.rlt' + '.jsonl'), 'a') as fw:
            path_ffn = []
            path = []
            base_path = []
            for tgt_layer in range(model.module.depth):
                if not tgt_layer:
                    # repeat for ig step
                    step_size = args.ig_total_step
                    x = x.repeat(step_size, 1, 1, 1)

                torch.cuda.synchronize()
                logits, ffn_weights = model(x, tgt_layer=tgt_layer)  # (b, n_cls), _, (b, ffn_size), 
                
                # x was repeated for ig step
                ffn_weights = ffn_weights[:args.batch_size]

                b_p = torch.argmax(ffn_weights.sum(dim=1), dim=-1)
                base_path.append(b_p.tolist())

                scaled_weights, weights_step = scaled_input(ffn_weights, args.ig_total_step)  # (num_points * bs, ffn_size), (bs, ffn_size)
                
                path_scaled_weights, path_weights_steps = path_scaled_input(path_ffn, path, args.ig_total_step)  
                scaled_weights.requires_grad_(True)
                
                for w in path_scaled_weights:
                    w.requires_grad_(True)
                
                path_scaled_weights.append(scaled_weights)
                path_weights_steps.append(weights_step)

                # integrated grad at the pred label for each layer
                ig_pred = []

                batch_weights = path_scaled_weights
                
                if not tgt_layer:
                    # repeat for ig step
                    step_size = batch_weights[-1].shape[0] // x.shape[0]
                    x = x.repeat(step_size, 1, 1, 1)

                _, grad = model(x, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=int(pred_label))  # (step * batch, ffn_size)

                for g in grad:
                    g = g.reshape(args.ig_total_step, args.batch_size, *g.shape[-2:]) # (m, b, ffn_size)
                    ig_pred.append(g)
                
                ja = ig_pred[-1]
                s = path_weights_steps[-1]

                for l, p in enumerate(path):
                    temp = torch.zeros_like(ig_pred[l][:, :, :, [0]])
                    for i, b in enumerate(p):
                        temp[:, i] = ig_pred[l][:, i, :, [b]]
                    ja = ja + temp
                
                # sum for ig steps
                ja = ja.sum(dim=0) 

                for l, p in enumerate(path):
                    temp_s = torch.zeros_like(path_weights_steps[l][:, :, [0]])
                    for i, b in enumerate(p):
                        temp_s[i] = path_weights_steps[l][i, :, [b]]
                    s = s + temp_s

                ja = ja * s

                # sum for n_tokens
                ja = ja.sum(dim=1)
                
                ja_p = torch.argmax(ja, dim=-1)

                path.append(ja_p.tolist())
                path_ffn.append(ffn_weights)

            
            path_scaled_weights, path_weights_steps = path_scaled_input(path_ffn, path, args.ig_total_step)  
            for w in path_scaled_weights:
                w.requires_grad_(True)
            _, grad = model(x, tgt_layer=model.module.depth-1, tmp_score=path_scaled_weights, tgt_label=int(pred_label))  # (step * batch, ffn_size)
            score = j_a_score(args, path, grad, path_weights_steps)

            base_path_scaled_weights, base_path_weights_steps = path_scaled_input(path_ffn, base_path, args.ig_total_step)  
            for w in base_path_scaled_weights:
                w.requires_grad_(True)
            _, base_grad = model(x, tgt_layer=model.module.depth-1, tmp_score=base_path_scaled_weights, tgt_label=int(pred_label))  # (step * batch, ffn_size)
            base_score = j_a_score(args, base_path, base_grad, base_path_weights_steps)

            base_path = torch.tensor(base_path).permute(1, 0).tolist()
            path = torch.tensor(path).permute(1, 0).numpy().tolist()
            # mean score for global tokens score/196.
            base_score = (base_score/196.).detach().cpu().numpy().tolist()
            score = (score/196.).detach().cpu().numpy().tolist()
            
            for b_s, b, p_s, p in zip(base_score, base_path, score, path):
                res_dict = {
                    'ja': [],
                    'ja_score': [],
                    'base': [],
                    'base_score':[]
                }
                res_dict['base_score'].append(b_s)
                res_dict['base'].append(b)
                res_dict['ja_score'].append(p_s)
                res_dict['ja'].append(p)
                res_dict_bag.append(res_dict)

            fw.write(res_dict_bag)
            toc = time.perf_counter()
            print(f"***** Relation: {str(int(pred_label))} evaluated. Costing time: {toc - tic:0.4f} seconds *****")

if __name__ == "__main__":
    main()