import os
import sys
import math
import networkx as nx
import numpy as np
import ctypes
import copy

mylib = ctypes.cdll.LoadLibrary('./libdnn_avx.so')

c_float_pointer_type = ctypes.POINTER(ctypes.c_float)

npc = 0
def npc_path():
    global npc
    ret = './intermediate-1/layer_{}.npy'.format(npc)
    npc += 1
    return ret
def npc_n():
    return './intermediate-1/layer_{}.npy'.format(npc - 1)

class DnnInferenceEngine(object):
    def __init__(self, graph, debug):
        self.g = graph
        self.debug = debug

    def run(self, tin):
        self.g.in_node.set_input(tin)
        out = {}
        currents = [self.g.in_node]
        done = set()
        counter = 0
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
                if not isinstance(current, Input):
                    counter += 1
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
        self.kernel = np.ascontiguousarray(kernel).astype(np.float32)
        self.strides = strides
        self.result = np.zeros((batch, out_height, out_width, out_channels), dtype='float32')

        self.name = name

    def run(self):
        print(self.name)
        in_layer = np.pad(
                self.in_node.result, 
                [(0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)], 
                'constant')
        mylib.conv2d(
                in_layer.ctypes.data_as(c_float_pointer_type),
                self.kernel.ctypes.data_as(c_float_pointer_type), 
                self.result.ctypes.data_as(c_float_pointer_type),
                *map(ctypes.c_int, self.result.shape),
                *map(ctypes.c_int, in_layer.shape[1:]),
                *map(ctypes.c_int, self.kernel.shape[:2]),
                *map(ctypes.c_int, self.strides[1:3]))
        nl = np.load(npc_path())
        print(abs(self.result - nl).max())

class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        if not (biases.ndim == 1 and in_node.result.shape[-1] == biases.shape[0]):
            raise ValueError

        self.in_node = in_node

        self.biases = np.ascontiguousarray(biases).astype(np.float32)
        self.result = np.zeros(in_node.result.shape, dtype=np.dtype(np.float32, align=True))

        self.name = name

    def run(self):
        print(self.name)
        mylib.bias_add(
                self.in_node.result.ctypes.data_as(c_float_pointer_type), 
                self.biases.ctypes.data_as(c_float_pointer_type), 
                self.result.ctypes.data_as(c_float_pointer_type),
                *map(ctypes.c_int, self.result.shape))
        print(abs(self.result - np.load(npc_path())).max())

class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        if not (padding == 'SAME' or padding == 'VALID'):
            raise ValueError
        
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
        
    def run(self):
        print(self.name)
        in_layer = np.pad(
                self.in_node.result, 
                [(0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)], 
                'constant', constant_values=np.finfo('float32').min)
        mylib.max_pool2d(in_layer.ctypes.data_as(c_float_pointer_type),
                self.result.ctypes.data_as(c_float_pointer_type),
                *self.result.shape,
                *in_layer.shape[1:],
                *self.ksize[1:3],
                *self.strides[1:3],
                self.pad_top, self.pad_bottom, self.pad_left, self.pad_right)

        nl = np.load(npc_path())
        print(abs(self.result - nl).max())

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
        

    def run(self):
        print(self.name)
        mylib.batch_norm(
                self.in_node.result.ctypes.data_as(c_float_pointer_type),
                self.mean.ctypes.data_as(c_float_pointer_type),
                self.variance.ctypes.data_as(c_float_pointer_type),
                self.gamma.ctypes.data_as(c_float_pointer_type),
                ctypes.c_float(self.epsilon),
                self.result.ctypes.data_as(c_float_pointer_type),
                *map(ctypes.c_int, self.result.shape))

        print(abs(self.result - np.load(npc_path())).max())


leaky_relu_vfunc = np.vectorize(
        lambda t: 0.1 * t if t < 0 else t)

class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        self.in_node = in_node
        self.result = np.zeros(in_node.result.shape, dtype='float32')

        self.name = name

    def run(self):
        print(self.name)
        mylib.leaky_relu(self.in_node.result.ctypes.data_as(c_float_pointer_type),
                self.result.ctypes.data_as(c_float_pointer_type),
                *map(ctypes.c_int, self.result.shape))

        print(abs(self.result - np.load(npc_path())).max())


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
