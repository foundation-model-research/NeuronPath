import itertools
import logging
import argparse
import math
import os
import re
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time
import sys

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

sys.path.append('../')
from models.modeling import VisionTransformer, CONFIGS

import torch.nn.functional as F

from tqdm import tqdm
from pprint import pprint

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def scaled_input(emb, ig_total_step):
    baseline = torch.zeros_like(emb)  
    num_points = ig_total_step
    step = (emb - baseline) / num_points  
    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0) 
    return res, step

def path_scaled_input(ffn_path, path, ig_total_step):
    steps = []
    ress = []
    for l, emb in enumerate(ffn_path):
        p = path[l]
        baseline = emb.clone()
        for i, b in enumerate(p):
            baseline[0, b] = 0 
        num_points = ig_total_step
        step = (emb - baseline) / num_points  
        res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0) 
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
            i=0
            temp[:, i] = ig_pred[l][:, i, [b]]
        ja = temp if ja is None else ja + temp
        
    ja = ja.sum(dim=0)

    for l, p in enumerate(path):
        temp_s = torch.zeros_like(step[l][:,[0]])
        for i, b in enumerate(p):
            i=0
            temp_s[i] = step[l][i, [b]]
        s = temp_s if s is None else s + temp_s
    ja = ja * s

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
                        default='../results_topk/',
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

    parser.add_argument("--ig_total_step",
                        default=20,
                        type=int,
                        help="Total step for cut.")
    parser.add_argument("--batch_size", choices=[1],
                        default=1,
                        type=int,
                        help="batch size")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_32"],
                        default="ViT-B_16",
                        help="Which model to use.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet1k"], default="imagenet1k")
    parser.add_argument("--dataset_path", default="../dataset/imagenet2012/val")
    parser.add_argument("--pretrain_weight", type=str, default='../ckpt/ViT-B_16-224.npz',
                        help="path to the pretrain weight")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")   

    parser.add_argument("--topK", default=5, type=int,
                        help="Choose Top K Neuron(s)")    
    # parse arguments
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass

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
    model.to(device)

    # data parallel
    if n_gpu > 1:
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
                             num_workers=4,
                             prefetch_factor=4,
                             pin_memory=True) if testset is not None else None

    epoch_iterator = tqdm(test_loader,
                          desc="Imgnet",
                          bar_format="{l_bar} {r_bar}",
                          dynamic_ncols=True,
                          )
    skipCounter=0

    def get_latest_pred_label_index(output_dir, output_prefix):
        pattern = re.compile(rf'{re.escape(output_prefix)}-(\d+)\.rlt\.jsonl')

        max_index = -1

        for file_name in os.listdir(output_dir):
            match = pattern.match(file_name)
            if match:
                pred_label = int(match.group(1))
                if pred_label > max_index:
                    max_index = pred_label

        return max_index

    skipCounter = get_latest_pred_label_index(args.output_dir, args.output_prefix)
    if skipCounter<=0:
        skipCounter=0
    else:
        # the last file may contain only part data
        # for simple just del it  
        outputjsonFileUrl=os.path.join(args.output_dir, args.output_prefix + '-' + str(int(skipCounter)) + '.rlt' + '.jsonl')
        os.remove(outputjsonFileUrl)
        skipCounter-=1
        skipCounter*=50
    

    print(f"should skip {skipCounter} times")
    for _, batch in enumerate(itertools.islice(epoch_iterator, skipCounter, None)):
        tic = time.perf_counter()
        batch = tuple(t.to(device) for t in batch)
        x, y = batch 
        pred_label = y[0]

        outputjsonFileUrl=os.path.join(args.output_dir, args.output_prefix + '-' + str(int(pred_label)) + '.rlt' + '.jsonl')

        with jsonlines.open(outputjsonFileUrl, 'a') as fw:

            path_ffn = []
            path = []
            base_path = []
            for tgt_layer in range(model.config.transformer.num_layers):
                logits, _, ffn_weights = model(x, tgt_layer=tgt_layer)  
                _, _indices = torch.topk(ffn_weights, args.topK, dim=-1)
                b_p=_indices.squeeze()

                base_path.append(b_p.tolist())
                scaled_weights, weights_step = scaled_input(ffn_weights, args.ig_total_step)  
                path_scaled_weights, path_weights_steps = path_scaled_input(path_ffn, path, args.ig_total_step)  

                scaled_weights.requires_grad_(True)
                
                for w in path_scaled_weights:
                    w.requires_grad_(True)
                
                path_scaled_weights.append(scaled_weights)
                path_weights_steps.append(weights_step)

                ig_pred = []

                batch_weights = path_scaled_weights
                _, grad = model(x, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=pred_label)  # (step * batch, ffn_size)
                
                for g in grad:
                    g = g.reshape(args.ig_total_step, args.batch_size, -1) # (m, b, ffn_size)
                    ig_pred.append(g)
                ja = ig_pred[-1]
                s = path_weights_steps[-1]
                for l, p in enumerate(path):
                    temp = torch.zeros_like(ig_pred[l][:,:,[0]])
                    for i, b in enumerate(p):
                        i=0
                        temp[:, i] = ig_pred[l][:, i, [b]]
                    ja = ja + temp
                ja = ja.sum(dim=0)
                for l, p in enumerate(path):
                    temp_s = torch.zeros_like(path_weights_steps[l][:,[0]])
                    for i, b in enumerate(p):
                        i=0
                        temp_s[i] = path_weights_steps[l][i, [b]]
                    s = s + temp_s
                ja = ja * s
                

                _, _indices = torch.topk(ja, args.topK, dim=-1)
                ja_p=_indices.squeeze()

                path.append(ja_p.tolist())
                path_ffn.append(ffn_weights)

            path_scaled_weights, path_weights_steps = path_scaled_input(path_ffn, path, args.ig_total_step)  
            for w in path_scaled_weights:
                w.requires_grad_(True)
            _, grad = model(x, tgt_layer=model.config.transformer.num_layers-1, tmp_score=path_scaled_weights, tgt_label=pred_label)  # (step * batch, ffn_size)
            score = j_a_score(args, path, grad, path_weights_steps)

            base_path_scaled_weights, base_path_weights_steps = path_scaled_input(path_ffn, base_path, args.ig_total_step)  
            for w in base_path_scaled_weights:
                w.requires_grad_(True)
            _, base_grad = model(x, tgt_layer=model.config.transformer.num_layers-1, tmp_score=base_path_scaled_weights, tgt_label=pred_label)  # (step * batch, ffn_size)
            base_score = j_a_score(args, base_path, base_grad, base_path_weights_steps)
    
            base_path = torch.tensor(base_path).numpy().tolist()
            path = torch.tensor(path).numpy().tolist()
            base_score = base_score.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy().tolist()
            
            res_dict = {
                'ja': path,
                'ja_score': score,
                'base': base_path,
                'base_score':base_score
            }


            fw.write(res_dict)
            toc = time.perf_counter()
            print(f"***** Relation: {str(int(pred_label))} evaluated. Costing time: {toc - tic:0.4f} seconds *****")

if __name__ == "__main__":
    main()