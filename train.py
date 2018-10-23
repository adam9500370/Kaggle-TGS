import sys, os
import cv2
import torch
import argparse
import timeit
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict, poly_lr_scheduler, AverageMeter
from ptsemseg.loss import *
from ptsemseg.augmentations import *

torch.backends.cudnn.benchmark = True

def train(args):
    sd = args.seed
    r_pad = args.r_pad

    # Setup Augmentations
    data_aug = Compose([RandomHorizontallyFlip(),
                        RandomTranslateWithReflect(max_translation=20),
                        RandomSizedCrop(size=args.img_rows, change_ar=False, min_area=0.8**2),], is_random_aug=True)

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm, augmentations=data_aug, num_k_split=args.num_k_split, max_k_split=args.max_k_split, sd=sd, r_pad=r_pad)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm, num_k_split=args.num_k_split, max_k_split=args.max_k_split, sd=sd, r_pad=r_pad)

    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup Model
    model = get_model(args.arch, n_classes, version=args.dataset, f_scale=args.feature_scale)
    model.cuda()

    vgg19_model = torchvision.models.vgg19(pretrained=True).cuda() # pretrained VGG19 for topology-aware loss
    vgg19_conv1_2 = nn.Sequential(*list(vgg19_model.features.children())[:3])
    vgg19_conv2_2 = nn.Sequential(*list(vgg19_model.features.children())[:8])
    vgg19_conv3_4 = nn.Sequential(*list(vgg19_model.features.children())[:17])

    for m in [vgg19_conv1_2, vgg19_conv2_2, vgg19_conv3_4]:
        for param in m.parameters():
            param.requires_grad = False


    # Check if model has custom optimizer / loss
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
    else:
        milestones = [x for x in range(50, args.n_epoch, 50)]
        gamma = 0.5

        ##optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.l_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam([
                                        {'params': [p for name, p in model.named_parameters() if p.requires_grad and 'gum_x3' not in name and 'cbr_gum3' not in name]},
                                        {'params': [p for name, p in model.named_parameters() if 'gum_x3' in name or 'cbr_gum3' in name], 'lr': args.l_rate*1e-1},
                                     ], lr=args.l_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        if args.num_cycles > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch//args.num_cycles, eta_min=args.l_rate*1e-1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if hasattr(model, 'loss'):
        print('Using custom loss')
        loss_fn = model.loss
    else:
        loss_fn = cross_entropy2d

    start_epoch = 0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model_dict = model.state_dict()
            model_dict.update(convert_state_dict(checkpoint['model_state']))
            model.load_state_dict(model_dict)

            #if checkpoint.get('optimizer_state', None) is not None:
            #    optimizer.load_state_dict(checkpoint['optimizer_state'])
            #    start_epoch = checkpoint['epoch']

            print("Loaded checkpoint '{}' (epoch {}, map {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['map']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 


    best_map = -100.0
    best_epoch = -1
    for epoch in range(start_epoch, args.n_epoch):
        start_train_time = timeit.default_timer()

        if args.num_cycles > 0:
            scheduler.step(epoch % (args.n_epoch // args.num_cycles)) # Cosine Annealing with Restarts
        else:
            scheduler.step(epoch)

        model.train()
        for i, (images, labels, dp_labels, names) in enumerate(trainloader):
            optimizer.zero_grad()

            images = images.cuda()
            labels = labels.cuda()
            dp_labels = dp_labels.cuda()

            outputs, offsets = model(images)

            loss_seg = loss_fn(outputs, labels, lambda_ce=args.lambda_ce, lambda_lv=args.lambda_lv)

            # Calculate topology-aware loss
            prob = F.softmax(outputs[0], dim=1)[:, 1, :, :]

            y_in = labels.unsqueeze(1).repeat(1, 3, 1, 1).float()
            f_in = prob.unsqueeze(1).repeat(1, 3, 1, 1).float()

            y_1 = vgg19_conv1_2(y_in)
            f_1 = vgg19_conv1_2(f_in)
            y_2 = vgg19_conv2_2(y_in)
            f_2 = vgg19_conv2_2(f_in)
            y_3 = vgg19_conv3_4(y_in)
            f_3 = vgg19_conv3_4(f_in)

            loss_top = args.lambda_top * (F.mse_loss(f_1, y_1) + F.mse_loss(f_2, y_2) + F.mse_loss(f_3, y_3))

            loss = loss_seg + loss_top
            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Iter [%6d/%6d] Loss: %.4f/%.4f" % (epoch+1, args.n_epoch, i+1, len(trainloader), loss_seg, loss_top))

        map = AverageMeter()
        mean_loss_seg_val = AverageMeter()
        mean_loss_top_val = AverageMeter()
        mean_loss_offset_val = AverageMeter()
        model.eval()
        with torch.no_grad():
            for i_val, (images_val, labels_val, dp_labels_val, names_val) in enumerate(valloader):
                images_val = images_val.cuda()
                labels_val = labels_val.cuda()
                dp_labels_val = dp_labels_val.cuda()

                outputs_val, offsets = model(images_val)

                loss_seg_val = loss_fn(outputs_val, labels_val, lambda_ce=args.lambda_ce, lambda_lv=args.lambda_lv)
                mean_loss_seg_val.update(loss_seg_val)

                pred = outputs_val.max(1)[1]

                loss_offset_val = args.lambda_offset * offsets.abs().mean() # GUM grid offsets
                mean_loss_offset_val.update(loss_offset_val)

                pred = pred.cpu().numpy()
                gt = labels_val.cpu().numpy()

                running_metrics.update(gt, pred)

                map_val = running_metrics.comput_map(gt, pred)
                map.update(map_val.mean(), n=map_val.size)

        print('Mean average precision: {:.5f}'.format(map.avg))
        print('Mean val loss: {:.4f}/{:.4f}'.format(mean_loss_seg_val.avg, mean_loss_offset_val.avg))

        score, class_iou = running_metrics.get_scores()

        for k, v in score.items():
            print(k, v)

        for i in range(n_classes):
            print(i, class_iou[i])

        state = {'epoch': epoch+1,
                 'model_state': model.state_dict(),
                 #'optimizer_state' : optimizer.state_dict(),
                 'map': map.avg,}
        torch.save(state, "checkpoints/{}_{}_{}_{}-{}_model.pth".format(args.arch, args.dataset, epoch+1, args.num_k_split, args.max_k_split))
        if map.avg >= best_map:
            best_map = map.avg
            best_epoch = epoch+1
            torch.save(state, "checkpoints/{}_{}_best_{}-{}_model.pth".format(args.arch, args.dataset, args.num_k_split, args.max_k_split))

        elapsed_train_time = timeit.default_timer() - start_train_time
        print('Training time (epoch {0:5d}): {1:10.5f} seconds'.format(epoch+1, elapsed_train_time))

        running_metrics.reset()
        map.reset()

    print('best map: {}, epoch: {}'.format(best_map, best_epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='pspnet', 
                        help='Architecture to use [\'fcn8s, unet, segnet, pspnet, icnet, etc\']')
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

    parser.add_argument('--n_epoch', nargs='?', type=int, default=200, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=40, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9, 
                        help='Momentum')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-4, 
                        help='Weight Decay')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=2, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    parser.add_argument('--seed', nargs='?', type=int, default=1234, 
                        help='Random seed')
    parser.add_argument('--r_pad', nargs='?', type=int, default=14, 
                        help='Reflective center image padding')
    parser.add_argument('--num_cycles', nargs='?', type=int, default=0, 
                        help='Cosine Annealing Cyclic LR')
    parser.add_argument('--lambda_top', nargs='?', type=float, default=5e-2, 
                        help='Weight for topology-aware loss')
    parser.add_argument('--lambda_offset', nargs='?', type=float, default=1.0, 
                        help='Weight for guided upsampling grid offset loss')
    parser.add_argument('--lambda_ce', nargs='?', type=float, default=1.0, 
                        help='Weight for cross entropy loss')
    parser.add_argument('--lambda_lv', nargs='?', type=float, default=1.0, 
                        help='Weight for lovasz softmax loss')

    parser.add_argument('--num_k_split', nargs='?', type=int, default=1,
                        help='The K-th fold cross validation')
    parser.add_argument('--max_k_split', nargs='?', type=int, default=10,
                        help='The total K fold cross validation')

    args = parser.parse_args()
    print(args)
    train(args)
