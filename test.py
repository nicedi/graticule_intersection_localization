# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import chainer
import chainer.functions as F
from chainer import serializers, optimizers, Variable
from chainer.cuda import to_gpu, to_cpu
import chainer.computational_graph as c
from scipy.misc import imread, imsave, imshow
from train import cg_variable
from locnet import LocNet
import cv2
import matplotlib.pyplot as plt
import argparse


# 在测试集上，每幅图像，在四个角附近寻找目标，推算顶点坐标，对每个角的一组候选坐标的xy分量取中位数作为最终结果。
# 记录最终结果和ground-truth的偏差，分别计算四个角上的偏差均值和标准差。

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--type', type=str, default='borders')
    parser.add_argument('--image', type=str, default='origin')
    args = parser.parse_args()
    
    # load model
    model = LocNet()
    serializers.load_npz("your model", model)
    
    if chainer.cuda.available and args.gpu >= 0:
        model.to_gpu(args.gpu)
    
    # data
    np.random.seed(1)
    datapath = 'path to the topographic maps'
    pkl_file = 'dataset_scaled.pkl'
        
    with open(os.path.join(datapath, pkl_file), 'rb') as fh:
        dataset = pickle.load(fh)
        
    np.random.shuffle(dataset)
    valset = dataset[:10]
    testset = dataset[10:110]
    trainset = dataset[110:]
    
    """
    dataset format:
    [
        {'bbox': [{'height': 58.95,
       'width': 60.21,
       'x': 264.43,
       'y': 352.01},
      {'height': 60.99,
       'width': 61.34,
       'x': 3668.43,
       'y': 302.12},
      {'height': 61.70,
       'width': 59.92,
       'x': 298.93,
       'y': 3316.31},
      {'height': 59.92,
       'width': 62.05,
       'x': 3714.53,
       'y': 3270.92}],
     'filename': '0/eb46cbcc612d82ead82bc7941499fefd.jpg'}

    ]
    """

    sample_mean = np.array([235.72, 231.17, 214.13], dtype=np.float32)
    patch_mean = sample_mean
    
    # Testing start
    testsize = len(testset)
    slide_step = 30
    offset = 300
    patchsize = 224
    bbox_min_size = 30
    
    test_results = []
    for item in testset:
        print(item['filename'])
        im = cv2.imread(os.path.join(datapath, item['filename']))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        bbox = item['bbox']
        h, w, _ = im.shape
        img = np.transpose(im, (2, 0, 1))
        
        results = {'filename':item['filename'], 'results':[]}
        
        try:
            for i in range(4):
                box = bbox[i]
                startx = box['x'] - offset if box['x'] - offset >= 0 else 0
                endx = box['x'] + offset/4 if box['x'] + offset/4 <= w else w
                starty = box['y'] - offset if box['y'] - offset >= 0 else 0
                endy = box['y'] + offset/4 if box['y'] + offset/4 <= h else h
                
                gt_box = (box['x'], box['y'], box['x']+box['width'], box['y']+box['height']) # format [xmin, ymin, xmax, ymax]
                
                for px in range(int(startx), int(endx), slide_step):
                    for py in range(int(starty), int(endy), slide_step):
                        if py+patchsize > h or px+patchsize > w:
                            continue
                        patch_box = (px, py, px+patchsize-1, py+patchsize-1)
                        # true_label = make_true_label(box, patch_box)
                        
                        tmp = {'patch_box':patch_box, 'gt_box':gt_box}
                        
                        patch = np.empty((1, 3, patchsize, patchsize), dtype=np.float32)
                        patch[0,:,:,:] = img[:, py:py+patchsize, px:px+patchsize]
                        patch -= patch_mean[np.newaxis, :, np.newaxis, np.newaxis]
                        
                        x_batch = cg_variable(patch, args.gpu)

                        with chainer.using_config('cudnn_deterministic', True):
                            with chainer.using_config('autotune', True):
                                with chainer.using_config('train', False):
                                    with chainer.no_backprop_mode():
                                        y_prob = F.sigmoid(model(x_batch))
                                        
                        if args.gpu >= 0:
                            y_prob.to_cpu()
                        y = (y_prob.data > 0.5).astype(np.int32)
                        bx = y[0,:patchsize].nonzero()
                        by = y[0, patchsize:].nonzero()
                        if args.type == 'in-out':
                            if len(bx[0]) < 1 or len(by[0]) < 1:
                                tmp['pred_box'] = None
                                results['results'].append(tmp)
                                continue
                            bx1 = bx[0][0]
                            bx2 = bx[0][-1]
                            by1 = by[0][0]
                            by2 = by[0][-1]
                        elif args.type == 'borders':
                            if len(bx[0]) < 3 or len(by[0]) < 3 or np.diff(bx[0]).max() < 5: # single peak
                                tmp['pred_box'] = None
                                results['results'].append(tmp)
                                continue
                            
                            bx_center = int(np.mean(bx[0]))                            
                            bx_left = bx[0][bx[0]<bx_center]
                            bx_right = bx[0][bx[0]>bx_center]
                            # peak position
                            bx_lpeak_pos = np.argmax(y[0,:patchsize][bx_left])
                            bx1 = bx_left[bx_lpeak_pos]
                            bx_rpeak_pos = np.argmax(y[0,:patchsize][bx_right])
                            bx2 = bx_right[bx_rpeak_pos]
                            
                            by_center = int(np.mean(by[0]))
                            by_up = by[0][by[0]<by_center]
                            by_bottom = by[0][by[0]>by_center]
                            # peak position
                            by_upeak_pos = np.argmax(y[0,patchsize:][by_up])
                            by1 = by_up[by_upeak_pos]
                            by_dpeak_pos = np.argmax(y[0,patchsize:][by_bottom])
                            by2 = by_bottom[by_dpeak_pos]
                        
                        bbox_pos = (px+bx1, py+by1, px+bx2, py+by2)

                        tmp['pred_box'] = bbox_pos
                        results['results'].append(tmp)
        except:
            # handling issues
            # import pdb; pdb.set_trace()
            pass
        test_results.append(results)

    # save test results
    with open('test_results_%s.pkl' % (args.type,), 'wb') as fh:
        pickle.dump(test_results, fh)
    