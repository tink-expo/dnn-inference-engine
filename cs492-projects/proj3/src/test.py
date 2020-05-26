import numpy as np
import scipy.signal
import math
import ctypes

c_float_pointer_type = ctypes.POINTER(ctypes.c_float)
c_int_pointer_type = ctypes.POINTER(ctypes.c_int)
mylib = ctypes.cdll.LoadLibrary('./libdnn_openblas.so')

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

def conv(im_shape, kernel_shape, strides):
    im = np.random.rand(*im_shape).astype(np.float32)
    kernel = np.random.rand(*kernel_shape).astype(np.float32)

    batch, ih, iw, ic = im_shape
    _, sh, sw, _ = strides
    kh, kw, _, od = kernel_shape

    oh, pad_top, pad_bottom = get_out_pads(ih, kh, sh, 'SAME')
    ow, pad_left, pad_right = get_out_pads(iw, kw, sw, 'SAME')

    im = np.pad(im, 
                [(0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)], 
                'constant')

    batch, ih, iw, ic = im.shape

    result1 = np.zeros((batch, oh, ow, od), dtype=np.float32)
    result2 = np.zeros(result1.shape, dtype=np.float32)
    result3 = np.zeros(result1.shape, dtype=np.float32)
    
    im = np.ascontiguousarray(im)
    kernel = np.ascontiguousarray(kernel)
    result1 = np.ascontiguousarray(result1)
    reuslt2 = np.ascontiguousarray(result2)
    reuslt3 = np.ascontiguousarray(result3)

    print(im[0,:,:,0])
    print(kernel[:,:,0,0])

    # 1. Scipy correlate2d
    for b in range(batch):
        for d in range(od):
            for c in range(ic):
                result1[b, :, :, d] += scipy.signal.correlate2d(im[b,:,:,c], kernel[:,:,c,d], mode='valid')

    # 2. mylib
    mylib.conv2d(
            im.ctypes.data_as(c_float_pointer_type),
            kernel.ctypes.data_as(c_float_pointer_type), 
            result2.ctypes.data_as(c_float_pointer_type),
            batch, oh, ow, od,
            ih, iw, ic,
            kh, kw,
            sh, sw)

    # 3. py loop
    for b in range(batch):
        for d in range(od):
            for c in range(ic):
                for i in range(oh):
                    for j in range(ow):
                        for di in range(kh):
                            for dj in range(kw):
                                result3[b, i, j, d] += (
                                    im[b, sh * i + di, sw * j + dj, c] *
                                    kernel[di, dj, c, d]
                                )

    return result1, result2, result3
        



r1, r2, r3 = conv((1, 3, 3, 1), (2, 2, 1, 1), strides=[1, 1, 1, 1])
print(r1[0,:,:,0])
print(r2[0,:,:,0])
print(r3[0,:,:,0])
print(abs(r1 - r2).max())