import sys, os
import cv2
import torch
import argparse
import timeit
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch.backends import cudnn
from torch.utils import data

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict, make_result_dir, AverageMeter
from ptsemseg.metrics import runningScore

cudnn.benchmark = True

def merge(args):
    result_root_path = make_result_dir(args.dataset, args.split)

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm, no_gt=args.no_gt)
    n_classes = loader.n_classes
    testloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    running_metrics = runningScore(n_classes)

    rm = 0
    pred_dict = {}
    map = AverageMeter()
    with open('list_test_18000') as f:
        id_list = f.read().splitlines()

    # Average all the probability maps from all folds
    all_prob = np.zeros((len(testloader), args.img_rows, args.img_cols), dtype=np.float32)
    for i in range(1, args.max_k_split+1):
        prob = np.load('prob-{}_{}_{}.npy'.format(args.split, i, args.max_k_split))
        all_prob = all_prob + prob
    all_prob = all_prob / args.max_k_split
    np.save('prob-{}_avg_{}.npy'.format(args.split, args.max_k_split), all_prob)

    for i, id in tqdm(enumerate(id_list)):
        lbl = id + '.png'
        pred = all_prob[i, :, :]

        if not args.no_gt:
            gt_path = os.path.join(data_path, args.split, 'masks', lbl)
            gt = loader.encode_segmap(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))

        pred = np.where(pred < args.pred_thr, 0, 1)

        if pred.sum() <= loader.lbl_thr: # Remove salt masks for sum of salts <= a threshold lbl_thr
            pred = np.zeros((args.img_rows, args.img_cols), dtype=np.uint8)
            rm = rm + 1

        decoded = loader.decode_segmap(pred)
        rle_mask = loader.RLenc(decoded)

        pred_dict[id] = rle_mask

        save_result_path = os.path.join(result_root_path, id + '.png')
        cv2.imwrite(save_result_path, decoded)

        if not args.no_gt:
            map_val = running_metrics.comput_map(gt, pred)
            map.update(map_val.mean(), n=map_val.size)

    if not args.no_gt:
        print('Mean Average Precision: {:.5f}'.format(map.avg))

    # Create final submission
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(args.split + '.csv')

    print('To black: ', rm)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='tgs', 
                        help='Dataset to use [\'pascal, camvid, ade20k, cityscapes, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=101, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=101, 
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test', 
                        help='Split of dataset to test on')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=2, 
                        help='Divider for # of features to use')

    parser.add_argument('--no_gt', dest='no_gt', action='store_true', 
                        help='Disable verification | True by default')
    parser.add_argument('--gt', dest='no_gt', action='store_false', 
                        help='Enable verification | True by default')
    parser.set_defaults(no_gt=True)

    parser.add_argument('--max_k_split', nargs='?', type=int, default=1, 
                        help='Total K-fold cross validation')

    parser.add_argument('--pred_thr', nargs='?', type=float, default=0.5,
                        help='Threshold of salt probability prediction')

    args = parser.parse_args()
    print(args)
    merge(args)
