"""
source: https://github.com/nmanchev/tptt/blob/master/tempOrder3bit.py
author: Hitesh Vaidya
"""

import numpy

class TempOrder3bitTask(object):
     def __init__(self, rng, floatX):
        self.rng = rng
        self.floatX = floatX
        self.nin = 6
        self.nout = 8
        self.classifType='lastSoftmax'

     def generate(self, batchsize, length):
        l = length
        p0 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.1)
        v0 = self.rng.randint(2, size=(batchsize,))
        p1 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.3)
        v1 = self.rng.randint(2, size=(batchsize,))
        p2 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.6)
        v2 = self.rng.randint(2, size=(batchsize,))
        targ_vals = v0 + v1*2 + v2 * 4
        vals  = self.rng.randint(4, size=(l, batchsize))+2
        vals[p0, numpy.arange(batchsize)] = v0
        vals[p1, numpy.arange(batchsize)] = v1
        vals[p2, numpy.arange(batchsize)] = v2
        data = numpy.zeros((l, batchsize, 6), dtype=self.floatX)
        targ = numpy.zeros((batchsize, 8), dtype=self.floatX)
        data.reshape((l*batchsize, 6))[numpy.arange(l*batchsize),
                                    vals.flatten()] = 1.
        targ[numpy.arange(batchsize), targ_vals] = 1.
        return data, targ