import sys, os
import cv2
import torch
import argparse
import timeit
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch.backends import cudnn
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict, make_result_dir, AverageMeter
from ptsemseg.metrics import runningScore

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, unary_from_softmax
except:
    print("Failed to import pydensecrf,\
           CRF post-processing will not work")

cudnn.benchmark = True

def test(args):
    sd = args.seed
    r_pad = args.r_pad

    result_root_path = make_result_dir(args.dataset, args.split)

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm, no_gt=args.no_gt, sd=sd, r_pad=r_pad, num_k_split=args.num_k_split, max_k_split=args.max_k_split)

    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)

    n_classes = loader.n_classes
    testloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Setup Model
    model = get_model(model_name, n_classes, version=args.dataset, f_scale=args.feature_scale)
    model.cuda()

    checkpoint = torch.load(args.model_path)
    state = convert_state_dict(checkpoint['model_state'])
    model_dict = model.state_dict()
    model_dict.update(state)
    model.load_state_dict(model_dict)

    print("Loaded checkpoint '{}' (epoch {}, map {})".format(args.model_path, checkpoint['epoch'], checkpoint['map']))

    running_metrics = runningScore(n_classes)

    rm = 0
    pred_dict = {}
    prob_dict = {}
    map = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, labels, dp_labels, names) in tqdm(enumerate(testloader)):
            images = images.cuda()
            images_flip = torch.from_numpy(np.copy(images.cpu().numpy()[:, :, :, ::-1])).cuda()#

            outputs = model(images)
            outputs_flip = model(images_flip)

            pred = F.softmax(outputs, dim=1)
            pred_flip = F.softmax(outputs_flip, dim=1)

            pred = pred.cpu().numpy()
            pred_flip = pred_flip.cpu().numpy()
            if not args.no_gt:
                gt = labels.numpy()

            pred = (pred[:, :, r_pad:-r_pad, r_pad:-r_pad] + pred_flip[:, :, :, ::-1][:, :, r_pad:-r_pad, r_pad:-r_pad]) / 2.0 if r_pad > 0 else (pred + pred_flip[:, :, :, ::-1]) / 2.0
            if not args.no_gt:
                gt = gt[:, r_pad:-r_pad, r_pad:-r_pad] if r_pad > 0 else gt

            if args.dcrf:
                images = images.cpu().numpy()
                n, c, h, w = pred.shape
                tmp = np.zeros((n, h, w), dtype=np.uint8)
                for k in range(pred.shape[0]):
                    unary = unary_from_softmax(pred[k])

                    """
                    img = images[k, :3, :, :].transpose(1, 2, 0)
                    img = img * 255. if args.img_norm else img
                    img = img + loader.mean
                    img = img.astype(np.uint8)
                    img = np.ascontiguousarray(img)
                    """

                    d = dcrf.DenseCRF2D(w, h, loader.n_classes)
                    d.setUnaryEnergy(unary)
                    #d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)
                    d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

                    q = d.inference(10)
                    tmp[k] = np.argmax(q, axis=0).reshape(h, w)
                pred = tmp
            else:
                #pred = np.argmax(pred, axis=1)
                pred = pred[:, 1, :, :]
                prob = np.copy(pred)
                pred = np.where(pred < args.pred_thr, 0, 1)

            for k in range(pred.shape[0]):
                if pred[k].sum() <= loader.lbl_thr:
                    pred[k] = np.zeros((args.img_rows, args.img_cols), dtype=np.uint8)
                    rm = rm + 1

            if not args.no_gt:
                running_metrics.update(gt, pred)

                map_val = running_metrics.comput_map(gt, pred)
                map.update(map_val.mean(), n=map_val.size)

            for k in range(pred.shape[0]):
                lbl = names[k][0]
                id = lbl.split('.')[0]

                decoded = loader.decode_segmap(pred[k])
                if decoded.shape[0] != 101 or decoded.shape[1] != 101:
                    decoded = cv2.resize(decoded, (101, 101), interpolation=cv2.INTER_NEAREST)#
                rle_mask = loader.RLenc(decoded)

                pred_dict[id] = rle_mask
                prob_dict[id] = prob[k]

                save_result_path = os.path.join(result_root_path, id + '_' + str(args.num_k_split) + '_' + str(args.max_k_split) + '.png')
                cv2.imwrite(save_result_path, decoded)

    if not args.no_gt:
        print('Mean Average Precision: {:.5f}'.format(map.avg))

        score, class_iou = running_metrics.get_scores()

        for k, v in score.items():
            print(k, v)

        for i in range(n_classes):
            print(i, class_iou[i])

        running_metrics.reset()
        map.reset()

    if args.split == 'test':
        with open('list_test_18000') as f:
            id_list = f.read().splitlines()

        all_prob = []
        for id in id_list:
            all_prob.append(np.expand_dims(prob_dict[id], axis=0))
        all_prob = np.concatenate(all_prob)
        np.save('prob-{}_{}_{}.npy'.format(args.split, args.num_k_split, args.max_k_split), all_prob)

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(args.split + '_' + str(args.num_k_split) + '_' + str(args.max_k_split) + '.csv')
    print('To black: ', rm)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='pspnet_101_cityscapes.pth', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k, cityscapes, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
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

    parser.add_argument('--dcrf', dest='dcrf', action='store_true', 
                        help='Enable DenseCRF based post-processing | False by default')
    parser.add_argument('--no-dcrf', dest='dcrf', action='store_false', 
                        help='Disable DenseCRF based post-processing | False by default')
    parser.set_defaults(dcrf=False)

    parser.add_argument('--seed', nargs='?', type=int, default=0, 
                        help='Random seed')
    parser.add_argument('--r_pad', nargs='?', type=int, default=0, 
                        help='Reflective center image padding')

    parser.add_argument('--num_k_split', nargs='?', type=int, default=0, 
                        help='K-th fold cross validation')
    parser.add_argument('--max_k_split', nargs='?', type=int, default=0, 
                        help='Total K fold cross validation')

    parser.add_argument('--pred_thr', nargs='?', type=float, default=0.5,
                        help='Threshold of salt probability prediction')

    args = parser.parse_args()
    print(args)
    test(args)
