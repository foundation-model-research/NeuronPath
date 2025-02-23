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

sys.path.append('../')
from models.modeling_ip import VisionTransformer, CONFIGS

import torch.nn.functional as F

from tqdm import tqdm

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
    parser.add_argument("--batch_size", choices=[1, 2, 5, 10, 25, 50],
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
    # model.train()

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

    for step, batch in enumerate(epoch_iterator):
        tic = time.perf_counter()
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        l = y[0]
        res_dict_bag = []

        with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '-' + str(int(l)) + '.rlt' + '.jsonl'), 'a') as fw:
            res_dict = {
                'path':[]
            }
            path = np.zeros((args.batch_size, model.config.transformer.num_layers)) - 1
            
            input_seq, _ = scaled_input(x, args.ig_total_step)
            
            _, ffn_weights = model(input_seq) 
            print(len(ffn_weights))
            for tgt_layer in range(model.config.transformer.num_layers):
                pred_label = l
                ig_pred = None
                
                _, grad = model(input_seq, tgt_layer=tgt_layer, tgt_label=pred_label, path=path, path_weights=ffn_weights[:(tgt_layer + 1)])  # (step * batch, ffn_size)
                
                new_p = torch.argmax(grad.mean(dim=0))
                path[:, tgt_layer] = new_p.detach().cpu().numpy()

            res_dict['path'] = path.tolist()

            fw.write(res_dict)
            toc = time.perf_counter()
            print(f"***** Relation: {str(int(l))} evaluated. Costing time: {toc - tic:0.4f} seconds *****")


if __name__ == "__main__":
    main()