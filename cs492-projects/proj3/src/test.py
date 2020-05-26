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

def im2col(im, oh, ow, kh, kw, sh, sw):
    ih, iw, ic = im.shape
    col = np.zeros((oh * ow, ic * kh * kw))

    for i in range(oh):
        for j in range(ow):
            patch = im[
                    i * sh : i * sh + kh,
                    j * sw : j * sw + kw,
                    :]
            for c in range(ic):
                col[i * ow + j, c * (kh * kw) : c * (kh * kw) + (kh * kw)] = \
                        patch[:, :, c].reshape(kh * kw)
    return col

def conv(im_shape, kernel_shape, strides):
    im = np.random.rand(*im_shape).astype(np.float32)
    kernel = np.random.rand(*kernel_shape).astype(np.float32)

    batch, ih, iw, ic = im_shape
    _, sh, sw, _ = strides
    kh, kw, kernel_ic, od = kernel_shape
    if ic != kernel_ic:
        raise ValueError

    oh, pad_top, pad_bottom = get_out_pads(ih, kh, sh, 'SAME')
    ow, pad_left, pad_right = get_out_pads(iw, kw, sw, 'SAME')

    im = np.pad(im, 
                [(0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)], 
                'constant')
    _, ih, iw, _ = im.shape

    result1 = np.zeros((batch, oh, ow, od), dtype=np.float32)
    result2 = np.zeros(result1.shape, dtype=np.float32)
    result3 = np.zeros(result1.shape, dtype=np.float32)
    
    im = np.ascontiguousarray(im)
    kernel = np.ascontiguousarray(kernel)
    result1 = np.ascontiguousarray(result1)
    reuslt2 = np.ascontiguousarray(result2)
    reuslt3 = np.ascontiguousarray(result3)

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

    # 3. matmul
    kernel_r = np.ascontiguousarray(
            kernel.transpose(2, 0, 1, 3).reshape(-1, od))
    for b in range(batch):
        imb = im[b, :, :, :]
        imcol = im2col(imb, oh, ow, kh, kw, sh, sw)
        mul = imcol.dot(kernel_r)
        result3[b, :, :, :] = mul.reshape(oh, ow, od)

    return result1, result2, result3
        



r1, r2, r3 = conv((1, 416, 416, 3), (3, 3, 3, 16), strides=[1, 1, 1, 1])
print(abs(r1 - r2).max())
print(abs(r1 - r3).max())