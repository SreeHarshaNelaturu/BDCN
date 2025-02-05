import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import time
import re
import os
import sys
import cv2
import ablation
from datasets.dataset import Data
import argparse
import cfg
from os import path as osp


def test(model, args):
    test_root = cfg.config_test[args.dataset]['data_root']
    test_lst = cfg.config_test[args.dataset]['data_lst']
    test_name_lst = os.path.join(test_root, test_lst)

    if 'Multicue' in args.dataset:
        test_lst = test_lst % args.k

    mean_bgr = np.array(cfg.config_test[args.dataset]['mean_bgr'])

    test_img = Data(test_root, test_lst, 0.5, mean_bgr=mean_bgr)
    testloader = torch.utils.data.DataLoader(
        test_img, batch_size=1, shuffle=False, num_workers=8)
    lst = np.loadtxt(test_name_lst, dtype=str)[:, 0]
    nm = [osp.splitext(osp.split(x)[-1])[0] for x in lst]
    save_dir = args.res_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.cuda:
        model.cuda()
    model.eval()
    data_iter = iter(testloader)
    iter_per_epoch = len(testloader)
    start_time = time.time()
    all_t = 0
    for i, (data, _) in enumerate(testloader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        t1 = time.time()
        out = model(data)
        t = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
        if not os.path.exists(os.path.join(save_dir, 'fuse')):
            os.mkdir(os.path.join(save_dir, 'fuse'))
        cv2.imwrite(os.path.join(save_dir, 'fuse', '%s.jpg'%nm[i]), 255-t*255)
        all_t += time.time() - t1

    print all_t
    print 'Overall Time use: ', time.time() - start_time

def main():
    import time
    print time.localtime()
    args = parse_args()
    args.bdcn = not args.no_bdcn
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = ablation.BDCN(ms=args.ms, block=args.block, bdcn=not args.no_bdcn,
        direction=args.dir, k=args.num_conv, rate=args.rate)
    model.load_state_dict(torch.load('%s' % (args.model)))

    test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
        default='bsds500', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='final-model/bdcn_pretrained_on_bsds500.pth',
        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='result',
        help='the dir to store result')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    parser.add_argument('--ms', action='store_true', default=False,
        help='whether employ the ms blocks, default False')
    parser.add_argument('--block', type=int, default=5,
        help='how many blocks of the model, default 5')
    parser.add_argument('--no-bdcn', action='store_true', default=False,
        help='whether to employ our policy to train the model, default False')
    parser.add_argument('--dir', type=str, choices=['both', 's2d', 'd2s'], default='both',
        help='the direction of cascade, default both')
    parser.add_argument('--num-conv', type=int, choices=[0,1,2,3,4], default=3,
        help='the number of convolution of SEB, default 3')
    parser.add_argument('--rate', type=int, default=4,
        help='the dilation rate of scale enhancement block, default 4')
    return parser.parse_args()

if __name__ == '__main__':
    main()
