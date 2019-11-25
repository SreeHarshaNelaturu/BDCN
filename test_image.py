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


def test(model):
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    start_time = time.time()
    all_t = 0
    data = cv2.imread("./datasets/BSR/BSDS500/data/images/test/279005.jpg")
    print("Working")
    data = np.array(data, np.float32)
    data -= mean_bgr
    data = data.transpose((2, 0, 1))
    data = torch.from_numpy(data).float().unsqueeze(0)
    if torch.cuda.is_available():
        data = data.cuda()
    data = Variable(data)
    t1 = time.time()
    out = model(data)
    
    out = [F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
    print("Complete")
    
    cv2.imwrite('./dank.png', 255-255*out[-1])
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
    test(model)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='final-model/bdcn_pretrained_on_bsds500.pth',
        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='result',
        help='the dir to store result')
    parser.add_argument('--data-root', type=str, default='./img')
    parser.add_argument('--test-lst', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    main()
