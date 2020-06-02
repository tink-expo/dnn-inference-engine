import ctypes
import numpy as np
import copy

mylib = ctypes.cdll.LoadLibrary('./mlib.so')

data = np.zeros((8), dtype=np.float32)
data.fill(5)
print(data)


# a = ctypes.c_float(24)
# b = ctypes.c_float(4.5)
# c = ctypes.c_float(0)

# mylib.add_float_p(a, b, ctypes.byref(c))
# print('(1) {} + {} = {}'.format(a.value, b.value, c.value))

# # Specify argtypes
# mylib.add_float_p.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
# mylib.add_float_p(a, b, c)
# print('(2) {} + {} = {}'.format(a.value, b.value, c.value))

# data = np.zeros((2, 3, 4), dtype=np.float32)


# count = 0
# for i in range(2):
#     for j in range(3):
#         for k in range(4):
#             data[i, j, k] = count
#             count += 1

# data = np.ascontiguousarray(data.transpose(2, 0, 1))
# for i in range(4):
#     for j in range(2):
#         for k in range(3):
#             print(data[i, j, k], end=" ")
# print()

data_p = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
# size = ctypes.c_int(data.size)
mylib.mf(data_p)

print(data)