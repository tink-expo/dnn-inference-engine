all: libdnn_cuda.so test_avx

libdnn_cuda.so: dnn_cuda.cu 
	nvcc -Xcompiler -fPIC -shared -o $@ $^

test_avx: avx_example.c
	gcc -o $@ $^ -mavx2 -L. -l:libdnn_cuda.so

clean:
	rm -f test_avx

