# all: libdnn_avx.so test_cuda

# libdnn_avx.so: dnn_avx.c
# 	gcc -shared -fPIC -o $@ $^ -mavx2

# test_cuda: cuda_example.cu
# 	nvcc -o $@ $^ -L/. -ldnn_avx

# clean:
# 	rm -f test_cuda libdnn_avx.so

test_cuda: test_cuda.o
	nvcc $^ -o $@

test_cuda.o: cuda_example.cu
	nvcc -c $< -o $@

clean:
	rm -f test_cuda test_cuda.o
