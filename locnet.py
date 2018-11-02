# -*- coding: utf-8 -*-
#import math
import chainer
import chainer.functions as F
import chainer.links as L

# The spatial size of the output volume can be computed as a function of the input volume size W:
# W' = (W - K + 2P)/S + 1

class BottleNeckA(chainer.Chain):   # adjusts number of channels
    def __init__(self, in_size, ch, out_size, stride=1):
        w = chainer.initializers.HeNormal()
        super(BottleNeckA, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, initialW=w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, initialW=w, nobias=True),
            # bn4=L.BatchNormalization(in_size),
            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, initialW=w, nobias=True),
        )

    def __call__(self, x):
        h1 = self.conv1(F.relu(self.bn1(x)))
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h1 = self.conv3(F.relu(self.bn3(h1)))
        h2 = self.conv4(x)
        return h1 + h2


class BottleNeckB(chainer.Chain):
    def __init__(self, in_size, ch):
        w = chainer.initializers.HeNormal()
        super(BottleNeckB, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, initialW=w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True),
            bn3=L.BatchNormalization(in_size),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, initialW=w, nobias=True),
        )

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.conv3(F.relu(self.bn3(h)))
        return h + x
    

class LocNet(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.HeNormal()
        self.insize = 224
        self.outsize = 14
        self.layers = {
            'ch1':110,
            'ch2':220,
            'ch3':330,
            'conv1':{'k':7, 's':2, 'p':3},
            'convX':{'k':(self.outsize, 1), 's':1, 'p':0},
            'convY':{'k':(1, self.outsize), 's':1, 'p':0}
        }
        super(LocNet, self).__init__(
            conv1=L.Convolution2D(3, self.layers['ch1'], self.layers['conv1']['k'], self.layers['conv1']['s'], self.layers['conv1']['p'], initialW=w, nobias=True),
            bn1=L.BatchNormalization(self.layers['ch1']),

            res1a=BottleNeckB(self.layers['ch1'], self.layers['ch1']),
            res1b=BottleNeckB(self.layers['ch1'], self.layers['ch1']),
            res1c=BottleNeckB(self.layers['ch1'], self.layers['ch1']),
            res1d=BottleNeckB(self.layers['ch1'], self.layers['ch1']),
            res2a=BottleNeckA(self.layers['ch1'], self.layers['ch1'], self.layers['ch2'], stride=2),
            res2b=BottleNeckB(self.layers['ch2'], self.layers['ch2']),
            res2c=BottleNeckB(self.layers['ch2'], self.layers['ch2']),
            res2d=BottleNeckB(self.layers['ch2'], self.layers['ch2']),
            res3a=BottleNeckA(self.layers['ch2'], self.layers['ch2'], self.layers['ch3'], stride=2),
            res3b=BottleNeckB(self.layers['ch3'], self.layers['ch3']),
            res3c=BottleNeckB(self.layers['ch3'], self.layers['ch3']),
            res3d=BottleNeckB(self.layers['ch3'], self.layers['ch3']),
            
            convX=L.Convolution2D(self.layers['ch3'], self.layers['ch3'], self.layers['convX']['k'], self.layers['convX']['s'], self.layers['convX']['p'], initialW=w, nobias=True),
            bnX=L.BatchNormalization(self.layers['ch3']),
            
            convY=L.Convolution2D(self.layers['ch3'], self.layers['ch3'], self.layers['convY']['k'], self.layers['convY']['s'], self.layers['convY']['p'], initialW=w, nobias=True),
            bnY=L.BatchNormalization(self.layers['ch3']),
            
            fcX1=L.Linear(self.layers['ch3']*self.outsize, 400),
            fcX2=L.Linear(400, self.insize),
            fcY1=L.Linear(self.layers['ch3']*self.outsize, 400),
            fcY2=L.Linear(400, self.insize),
        )

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.res1a(h)
        h = self.res1b(h)
        h = self.res1c(h)
        h = self.res1d(h)
        h = self.res2a(h)
        h = self.res2b(h)
        h = self.res2c(h)
        h = self.res2d(h)
        h = self.res3a(h)
        h = self.res3b(h)
        h = self.res3c(h)
        h = self.res3d(h)
        
        hX = self.convX(F.relu(self.bnX(h)))
        hX = self.fcX1(F.relu(hX))
        hX = self.fcX2(F.relu(hX))
        
        hY = self.convY(F.relu(self.bnY(h)))
        hY = self.fcY1(F.relu(hY))
        hY = self.fcY2(F.relu(hY))

        return F.concat((hX, hY))