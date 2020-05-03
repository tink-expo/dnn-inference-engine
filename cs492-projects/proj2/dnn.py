import os
import sys
import math
import networkx as nx
import numpy as np
import multiprocessing
from itertools import repeat

class DnnInferenceEngine(object):
    def __init__(self, graph):
        self.g = graph

    def run(self, tin):
        self.g.in_node.set_input(tin)
        out = {}
        currents = [self.g.in_node]
        done = set()
        while (len(currents) != 0):
            nexts = []
            for current in currents:
                skip_current = False
                predecessors = self.g.G.predecessors(current)
                for predecessor in predecessors:
                    if predecessor not in done:
                        nexts.append(predecessor)
                        skip_current = True
                if skip_current:
                    continue
                current.run()
                if self.g.is_out_node(current):
                    out = current.result
                done.add(current)
                for successor in self.g.G.successors(current):
                    nexts.append(successor)
            currents = nexts
        return out

class DnnGraphBuilder(object):
    def __init__(self):
        self.G = nx.DiGraph()
        self.name_num = {"conv2d": 0, 
                         "bias_add": 0, 
                         "max_pool2d": 0, 
                         "batch_norm": 0, 
                         "leaky_relu": 0, 
                         "input": 0}
        self.in_node = None
        self.out_node = None

    def set_in_node(self, node):
        self.in_node = node

    def set_out_node(self, node):
        self.out_node = node

    def is_out_node(self, node):
        return self.out_node is node

    def get_name(self, layer_name):
        name = layer_name + "_" + str(self.name_num[layer_name])
        self.name_num[layer_name] += 1
        return name

    def create_conv2d(self, in_node, kernel, strides, padding):
        out_node = Conv2D(self.get_name("conv2d"), in_node, kernel, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_bias_add(self, in_node, biases):
        out_node = BiasAdd(self.get_name("bias_add"), in_node, biases)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_max_pool2d(self, in_node, ksize, strides, padding):
        out_node = MaxPool2D(self.get_name("max_pool2d"), in_node, ksize, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_batch_norm(self, in_node, mean, variance, gamma, epsilon):
        out_node = BatchNorm(self.get_name("batch_norm"), in_node, mean, variance, gamma, epsilon)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_leaky_relu(self, in_node):
        out_node = LeakyReLU(self.get_name("leaky_relu"), in_node)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_input(self, in_shape):
        out_node = Input(self.get_name("input"), in_shape)
        self.G.add_node(out_node) 
        self.set_in_node(out_node)  # Assume there's only one input
        return out_node

class DnnNode(object):
    def __init__(self):
        pass

    def run(self):
        self.result = None 

#
# Complete below classes.
#

def get_out_pads(in_size, filter_size, stride_size, padding):
    assert padding == 'SAME' or padding == 'VALID'

    if padding == 'SAME':
        out_size = math.ceil(float(in_size) / float(stride_size))
        pad_size = max(
            (out_size - 1) * stride_size + filter_size - in_size, 0)
        pad_front = pad_size // 2
        pad_back = pad_size - pad_front

    else:
        out_size = math.ceil(float(in_size - filter_size + 1) / float(stride_size))
        pad_front = 0
        pad_back = 0

    return out_size, pad_front, pad_back

def pre_multiprocessing(batch, out_channels):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    bd_lst = []
    for b in range(batch):
        for d in range(out_channels):
            bd_lst.append((b, d))

    return pool, bd_lst

def post_multiprocessing(batch, out_channels, work_results, result):
    for b in range(batch):
        for d in range(out_channels):
            result[b, :, :, d] = work_results[b * out_channels + d]


def conv2d_work(bd, in_layer, kernel, strides, result_shape):
    b, d = bd
    filter_height, _, in_channels, _ = kernel.shape
    _, out_height, out_width, _ = result_shape
    res2d = np.zeros((out_height, out_width))
    for c in range(in_channels):
        for i in range(out_height):
            for di in range(filter_height):
                res2d[i, :] += (
                    np.correlate(
                            in_layer[b, strides[1] * i + di, :, c], 
                            kernel[di, :, c, d]))
    return res2d

class Conv2D(DnnNode):
    def __init__(self, name, in_node, kernel, strides, padding):
        batch, in_height, in_width, in_channels = in_node.result.shape
        filter_height, filter_width, filter_in_channels, out_channels = kernel.shape
        if filter_in_channels != in_channels:
            raise ValueError

        if not (padding == 'SAME' or padding == 'VALID'):
            raise ValueError

        out_height, self.pad_top, self.pad_bottom = get_out_pads(
                in_height, filter_height, strides[1], padding)
        out_width, self.pad_left, self.pad_right = get_out_pads(
                in_width, filter_width, strides[2], padding)

        self.in_node = in_node
        self.kernel = kernel
        self.strides = strides
        self.result = np.zeros((batch, out_height, out_width, out_channels), dtype='float32')

        self.name = name
        print(name)

    def run(self):
        in_layer = self.in_node.result
        in_layer = np.pad(
                in_layer, 
                [(0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)], 
                'constant')
        
        batch, out_height, out_width, out_channels = self.result.shape
        filter_height, filter_width, in_channels, _ = self.kernel.shape

        pool, bd_lst = pre_multiprocessing(batch, out_channels)

        work_results = pool.starmap(conv2d_work, zip(bd_lst, 
                repeat(in_layer), repeat(self.kernel), repeat(self.strides), repeat(self.result.shape)))
        pool.close()
        pool.join()

        post_multiprocessing(batch, out_channels, work_results, self.result)


def bias_add_work(bd, in_layer, bias):
    b, d = bd
    return in_layer[b, :, :, d] + bias

class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        if not (biases.ndim == 1 and in_node.result.shape[-1] == biases.shape[0]):
            raise ValueError

        self.in_node = in_node
        self.biases = biases
        self.result = np.zeros(in_node.result.shape, dtype='float32')

        self.name = name
        print(name)

    def run(self):
        batch, _, _, out_channels = self.result.shape
        pool, bd_lst = pre_multiprocessing(batch, out_channels)
        work_results = pool.starmap(bias_add_work, zip(bd_lst,
                repeat(self.in_node.result), self.biases))
        
        pool.close()
        pool.join()

        post_multiprocessing(batch, out_channels, work_results, self.result)


def max_pool2d_work(bd, in_layer, ksize, strides, result_shape):
    b, d = bd
    _, out_height, out_width, _ = result_shape
    res2d = np.zeros((out_height, out_width))
    for i in range(out_height):
        for j in range(out_width):
            in_i = i * strides[1]
            in_j = j * strides[2]
            res2d[i, j] = np.max(
                in_layer[b, in_i:in_i+ksize[1], in_j:in_j+ksize[2], d])
    return res2d

class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        batch, in_height, in_width, in_channels = in_node.result.shape
        out_height, self.pad_top, self.pad_bottom = get_out_pads(
                in_height, ksize[1], strides[1], padding)
        out_width, self.pad_left, self.pad_right = get_out_pads(
                in_width, ksize[2], strides[2], padding)

        self.in_node = in_node
        self.ksize = ksize
        self.strides = strides
        self.result = np.zeros((batch, out_height, out_width, in_channels), dtype='float32')

        self.name = name
        print(name)
        
    def run(self):
        in_layer = self.in_node.result
        in_layer = np.pad(
                in_layer, 
                [(0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)], 
                'constant', constant_values=np.finfo('float32').min)
        
        batch, out_height, out_width, out_channels = self.result.shape

        pool, bd_lst = pre_multiprocessing(batch, out_channels)
        work_results = pool.starmap(max_pool2d_work, zip(bd_lst,
                repeat(in_layer), repeat(self.ksize), repeat(self.strides), repeat(self.result.shape)))
        
        pool.close()
        pool.join()

        post_multiprocessing(batch, out_channels, work_results, self.result)


def batch_norm_work(bd, in_layer, mean, variance, gamma, epsilon):
    b, d = bd
    return ((in_layer[b, :, :, d] - mean) / np.sqrt(variance + epsilon)) * gamma

class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        if not all(arg.ndim == 1 and in_node.result.shape[-1] == arg.shape[0] 
                for arg in [mean, variance, gamma]):
            raise ValueError

        self.in_node = in_node
        self.mean = mean
        self.variance = variance
        self.gamma = gamma
        self.epsilon = epsilon
        self.result = np.zeros(in_node.result.shape, dtype='float32')

        self.name = name
        print(name)
        

    def run(self):
        batch, _, _, out_channels = self.result.shape
        pool, bd_lst = pre_multiprocessing(batch, out_channels)
        work_results = pool.starmap(batch_norm_work, zip(bd_lst,
                repeat(self.in_node.result), self.mean, self.variance, self.gamma, repeat(self.epsilon)))
        
        pool.close()
        pool.join()

        post_multiprocessing(batch, out_channels, work_results, self.result)


leaky_relu_vfunc = np.vectorize(
        lambda t: 0.1 * t if t < 0 else t)

def leaky_relu_work(bd, in_layer):
    b, d = bd
    return leaky_relu_vfunc(in_layer[b, :, :, d])

class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        self.in_node = in_node
        self.result = np.zeros(in_node.result.shape, dtype='float32')

        func = lambda t: 0.1 * t if t < 0 else t
        self.vfunc = np.vectorize(func)

        self.name = name
        print(name)

    def run(self):
        batch, _, _, out_channels = self.result.shape
        pool, bd_lst = pre_multiprocessing(batch, out_channels)
        work_results = pool.starmap(leaky_relu_work, zip(bd_lst,
                repeat(self.in_node.result)))

        pool.close()
        pool.join()

        post_multiprocessing(batch, out_channels, work_results, self.result)



# Do not modify below
class Input(DnnNode):
    def __init__(self, name, in_shape):
        self.name = name
        self.in_shape = in_shape 
        self.result = np.ndarray(self.in_shape)

    def set_input(self, tensor):
        assert tuple(self.in_shape) == tuple(tensor.shape)
        self.result = tensor 

    def run(self):
        pass

