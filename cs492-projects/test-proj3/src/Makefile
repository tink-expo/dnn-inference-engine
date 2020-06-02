all: libdnn_openblas.so libdnn_avx.so libdnn_cublas.so libdnn_cuda.so

libdnn_openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^ -lopenblas

libdnn_avx.so: dnn_avx.c
	gcc -shared -fPIC -O3 -o $@ $^ -mavx2
#	gcc -shared -fPIC -o $@ $^ -mavx2

libdnn_cublas.so: dnn_cublas.cu 
	nvcc -Xcompiler -fPIC -shared -o $@ $^ -lcublas

libdnn_cuda.so: dnn_cuda.cu 
#	nvcc -Xcompiler -fPIC -Xcompiler -mavx -Xcompiler -O3 -shared -o $@ $^
	nvcc -Xcompiler -fPIC -Xcompiler -mavx2 -shared -o $@ $^

clean:
	rm -f *.so
