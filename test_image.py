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
import bdcn
from datasets.dataset import Data
import argparse
import cfg
from matplotlib import pyplot as plt
import os
import os.path as osp
from scipy.io import savemat


def make_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def test(model, args):
    test_root = args.data_root
    save_dir = args.res_dir
    if args.test_lst is not None:
        with open(osp.join(test_root, args.test_lst), 'r') as f:
            test_lst = f.readlines()
        test_lst = [x.strip() for x in test_lst]
        if ' ' in test_lst[0]:
            test_lst = [x.split(' ')[0] for x in test_lst]
    else:
        test_lst = os.listdir(test_root)
    #print(test_lst[0])
    save_sideouts = 1
    k = 1
    if save_sideouts:
        for j in xrange(5):
            make_dir(os.path.join(save_dir, 's2d_'+str(k)))
            make_dir(os.path.join(save_dir, 'd2s_'+str(k)))
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.cuda:
        model.cuda()
    model.eval()
    start_time = time.time()
    all_t = 0
    data = cv2.imread("horn.jpg")
    print("hubba")
    data = np.array(data, np.float32)
    data -= mean_bgr
    data = data.transpose((2, 0, 1))
    data = torch.from_numpy(data).float().unsqueeze(0)
    if args.cuda:
        data = data.cuda()
    data = Variable(data)
    t1 = time.time()
    out = model(data)
    if save_sideouts:
        out = [F.sigmoid(x).cpu().data.numpy()[0, 0, :, :] for x in out]
        #k = 1
        #for j in xrange(5):
        #    cv2.imwrite(os.path.join(save_dir, 's2d_'+str(k), "%s.jpg"%j), 255-t*255)
        #    cv2.imwrite(os.path.join(save_dir, 'd2s_'+str(k), '%s.jpg'%j), 255-255*t)
        #    k += 1
    else:
        out = [F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
    if not os.path.exists(os.path.join(save_dir, 'fuse')):
        os.mkdir(os.path.join(save_dir, 'fuse'))
    cv2.imwrite(os.path.join(save_dir, 'fuse/dank.png'), 255*out[-1])
    all_t += time.time() - t1
    print all_t
    print 'Overall Time use: ', time.time() - start_time

def main():
    import time
    print time.localtime()
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = bdcn.BDCN()
    model.load_state_dict(torch.load('%s' % (args.model), map_location='cpu'))
    print model.fuse.weight.data
    test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='final_model/bdcn_10000.pth',
        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='result',
        help='the dir to store result')
    parser.add_argument('--data-root', type=str, default='./img')
    parser.add_argument('--test-lst', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    main()
