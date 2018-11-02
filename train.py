# -*- coding: utf-8 -*-
import os, pickle
import numpy as np
import chainer
import chainer.functions as F
#import chainer.links as L
from chainer import Variable, serializers, optimizers
from chainer.cuda import to_gpu, to_cpu
# from scipy.misc import imread, imsave, imshow
import argparse
from locnet import LocNet
from visdom import Visdom
import matplotlib.pyplot as plt
from multiprocessing import Queue, Process
import cv2


# 注意运用keepdims参数: some_array.sum(axis=0, keepdims=True)

def cg_variable(obj, gpu):
    if chainer.cuda.available and gpu >= 0:
        return chainer.Variable(chainer.cuda.to_gpu(obj))
    else: 
        return chainer.Variable(obj)
    
    
def save_model_state(model, optimizer, filename):
    print('save the model')
    serializers.save_npz("%s.model" % (filename,), model)
    print('save the optimizer')
    serializers.save_npz("%s.state" % (filename,), optimizer)
    
    
def load_model_state(model, optimizer, filename):
    print('load the model')
    serializers.load_npz("%s.model" % (filename,), model)
    print('load the optimizer')
    serializers.load_npz("%s.state" % (filename,), optimizer)
    
    
def gen_batch(items, batchsize, patchsize, type='in-out'):
    item_count = len(items)
    x_batch = np.empty((batchsize*item_count, 3, patchsize, patchsize), dtype=np.float32)
    y_batch = np.zeros((batchsize*item_count, patchsize*2), dtype=np.float32)
    for j in range(item_count):
        item = items[j]
        im = cv2.imread(os.path.join(datapath, item['filename'])) 
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape
        img = np.transpose(im, (2, 0, 1))
        
        neg_samples = np.zeros((batchsize//2, 2), dtype=np.float32)
        neg_samples[0 : batchsize//4, 0] = np.random.choice(
                        np.arange(0, w-patchsize), size=batchsize//4, replace=False)
        neg_samples[0 : batchsize//4, 1] = np.random.choice(
                        np.arange(item['bbox'][0]['y']+70, item['bbox'][2]['y']-patchsize), size=batchsize//4, replace=False)
        neg_samples[batchsize//4 : batchsize//2, 0] = np.random.choice(
                        np.arange(item['bbox'][0]['x']+70, item['bbox'][1]['x']-patchsize), size=batchsize//4, replace=False)
        neg_samples[batchsize//4 : batchsize//2, 1] = np.random.choice(
                        np.arange(0, h-patchsize), size=batchsize//4, replace=False)
        
        for k in range(batchsize//2):
            x_batch[j*batchsize+k] = img[:, int(neg_samples[k,1]):int(neg_samples[k,1]+patchsize), int(neg_samples[k,0]):int(neg_samples[k,0]+patchsize)]
        
        pos_samples = np.zeros((batchsize//2, 6+patchsize*2), dtype=np.float32)
        
        for i in range(4):
            bbox = item['bbox'][i]
            pos_samples[i*batchsize//8 : (i+1)*batchsize//8, 0] = np.random.choice(
                            np.arange(0 if bbox['x']-160<0 else bbox['x']-160, bbox['x'] if bbox['x']+patchsize<w else w-patchsize), size=batchsize//8, replace=False)
            pos_samples[i*batchsize//8 : (i+1)*batchsize//8, 1] = np.random.choice(
                            np.arange(0 if bbox['y']-160<0 else bbox['y']-160, bbox['y'] if bbox['y']+patchsize<h else h-patchsize), size=batchsize//8, replace=False)
            pos_samples[i*batchsize//8 : (i+1)*batchsize//8, 2] = bbox['x']
            pos_samples[i*batchsize//8 : (i+1)*batchsize//8, 3] = bbox['y']
            pos_samples[i*batchsize//8 : (i+1)*batchsize//8, 4] = bbox['width']
            pos_samples[i*batchsize//8 : (i+1)*batchsize//8, 5] = bbox['height']
        
        # data format: (patchX, patchY, bboxX, bboxY, bboxW, bboxH, Tx, Ty)
        if type == 'in-out':
        # calculate T
            for i in range(batchsize//2):
                pos_samples[i, int(6+pos_samples[i,2]-pos_samples[i,0]) : int(6+pos_samples[i,2]-pos_samples[i,0]+pos_samples[i,4])] = 1
                pos_samples[i, int(6+patchsize+pos_samples[i,3]-pos_samples[i,1]) : int(6+patchsize+pos_samples[i,3]-pos_samples[i,1]+pos_samples[i,5])] = 1
        elif type == 'borders':
            try:
                for i in range(batchsize//2):
                    pos_samples[i, int(6+pos_samples[i,2]-pos_samples[i,0])] = 1
                    pos_samples[i, int(6+pos_samples[i,2]-pos_samples[i,0]+pos_samples[i,4])] = 1
                    pos_samples[i, int(6+patchsize+pos_samples[i,3]-pos_samples[i,1])] = 1
                    pos_samples[i, int(6+patchsize+pos_samples[i,3]-pos_samples[i,1]+pos_samples[i,5])] = 1
            except IndexError:
                pos_samples[i, -1] = 1
        else:
            try:
                for i in range(batchsize//2):
                    pos_samples[i, int(6+pos_samples[i,2]-pos_samples[i,0])] = 1
                    pos_samples[i, int(6+patchsize+pos_samples[i,3]-pos_samples[i,1])] = 1
            except IndexError:
                pos_samples[i, -1] = 1
            
            
        for k in range(batchsize//2):
            x_batch[j*batchsize+batchsize//2+k] = img[:, int(pos_samples[k,1]):int(pos_samples[k,1]+patchsize), int(pos_samples[k,0]): int(pos_samples[k,0]+patchsize)]
            y_batch[j*batchsize+batchsize//2+k] = pos_samples[k, 6:]     
    
    # mean normalization
    x_batch -= patch_mean[np.newaxis, :, np.newaxis, np.newaxis]
    
    # data permutation
    perm_idx = np.random.permutation(np.arange(batchsize*item_count))
    return x_batch[perm_idx], y_batch[perm_idx]


def evaluator(dataset, gpu, type='in-out'):
    st = np.random.get_state()
    np.random.seed(101)
    patchsize = 224
    n_epoch = 5
    total_loss = 0
    for epoch in range(n_epoch):
        np.random.shuffle(dataset)
        for idx in range(0, len(dataset), 2):
            items = dataset[idx : idx+2] # sample from two images
            x_batch, t = gen_batch(items, batchsize, patchsize, type=type)
            x_batch = cg_variable(x_batch, gpu)
            
            if type == 'in-out':
                t = cg_variable(t, gpu)
            elif type == 'borders':
                tP = cg_variable(patchsize*t/2, gpu)
                tN = cg_variable(0.5*(1-t), gpu)
            else:
                tP = cg_variable(patchsize*t, gpu)
                tN = cg_variable(1-t, gpu)
                
            with chainer.using_config('cudnn_deterministic', True):
                with chainer.using_config('autotune', True):
                    with chainer.using_config('train', False):
                        with chainer.no_backprop_mode():
                            y = model(x_batch)
            if type == 'in-out':
                loss = F.bernoulli_nll(t, y)
            else:
                loss = -F.sum(tP*F.log(F.clip(F.sigmoid(y), 1e-10, 1.0)) + tN*F.log(1-F.clip(F.sigmoid(y), 1e-10, 1-1e-10)))
            total_loss += loss.data
    np.random.set_state(st)
    return total_loss / n_epoch / len(dataset)


def plot_training_loss(loss, iteration):
    step = np.ones(1) * iteration
    viz.line(
        X = step,
        Y = loss,
        win = win,
        update = 'append',
    )
    
    
# 0. prepare the model and data
# 1. read 2 images, sample positive and negative image patches to build training batch
# 2. calculate target probabilities
# 3. feed into the model and optimize it

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--type', type=str, default='in-out')
    parser.add_argument('--image', type=str, default='origin')
    args = parser.parse_args()
    
    # visualize training process
    viz = Visdom()
    assert viz.check_connection()
    win = viz.line(
        X = np.array([0]),
        Y = np.zeros((1)),
        opts = dict(
            xlabel = 'Iteration',
            ylabel = 'Loss',
            title = 'Training Loss',
            legend = ['Train Loss']
        )
    )
    
    # model and optimizer
    # Border model use RMSprop(lr=0.0001) with WeightDecay(0.0005), it is more hard to train than the In-Out model.
    model = LocNet()
    optimizer = optimizers.RMSprop(lr=0.0001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
    
    # count params
    param_count = 0
    for param in model.params():
        param_count += param.data.size
    print('parameters count: %d' % (param_count,))
    
    if chainer.cuda.available and args.gpu >= 0:
        model.to_gpu(args.gpu)
    
    # data
    savehead = 'locnet'+args.type
    np.random.seed(1)
    datapath = 'your path to topographic maps'
    
    # the following .pkl file has the ground truth
    pkl_file = 'dataset_scaled.pkl'
        
    with open(os.path.join(datapath,pkl_file), 'rb') as fh:
        dataset = pickle.load(fh)
        
    np.random.shuffle(dataset)
    valset = dataset[:10]
    testset = dataset[10:110]
    trainset = dataset[110:]

    sample_mean = np.array([235.72, 231.17, 214.13], dtype=np.float32)
    patch_mean = sample_mean
    
    # Training start
    trainsize = len(trainset)
    n_epoch = 3
    start_idx = 0
    batchsize = 8 # batch per image
    patchsize = 224
    best_train_loss = np.Inf
    best_test_loss = np.Inf
    
    queue = Queue(50)
    
    def sample_feeder(queue):
        for epoch in range(n_epoch):
            np.random.shuffle(trainset)
            for idx in range(0, trainsize):
                items = trainset[idx : idx+2]
                x_data, y_data = gen_batch(items, batchsize, patchsize, type=args.type)
                
                queue.put((x_data, y_data, epoch, idx, 'TRAIN'))
        queue.put((None, None, None, None, 'STOP'))
        
    feeder_proc = Process(target=sample_feeder, args=(queue,))
    feeder_proc.start()
    

    while True:
        x_data, t, epoch, idx, flag = queue.get()
        if flag == 'STOP':
            break
        
        x_data = cg_variable(x_data, args.gpu)
        if args.type == 'in-out':
            t = cg_variable(t, args.gpu)
        elif args.type == 'borders':
            tP = cg_variable(patchsize*t/2, args.gpu)
            tN = cg_variable(0.5*(1-t), args.gpu)
        else:
            tP = cg_variable(patchsize*t, args.gpu)
            tN = cg_variable(1-t, args.gpu)

        with chainer.using_config('cudnn_deterministic', True):
            with chainer.using_config('autotune', True):
                y = model(x_data)
                
        if args.type == 'in-out':
            loss = F.bernoulli_nll(t, y)
        else:
            loss = -F.sum(tP*F.log(F.clip(F.sigmoid(y), 1e-10, 1.0)) + tN*F.log(1-F.clip(F.sigmoid(y), 1e-10, 1-1e-10)))
        
        current_loss = to_cpu(loss.data).flatten()
        step = epoch*trainsize+idx+1
        plot_training_loss(current_loss, step)
        
        # optimize the model
        info = "epoch: %d, idx: %d, loss: %.4f" % (epoch, idx, current_loss)
        print(info)
        
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        # evaluate the model
        if epoch > 1 and current_loss < best_train_loss or epoch > 1 and idx % 10 == 0:
            if current_loss < best_train_loss:
                best_train_loss = current_loss
                
            test_loss = evaluator(valset, args.gpu, type=args.type)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                # save the model
                save_model_state(model, optimizer, "%s_validationLoss_%d" % (savehead, test_loss))
            test_info = "epoch: %d, idx: %d, test loss: %.4f" % (epoch, idx, test_loss)
            print(test_info)
        
    queue.close()
    queue.join_thread()
    feeder_proc.join()