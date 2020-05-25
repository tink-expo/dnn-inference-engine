all: libdnn_openblas.so libdnn_avx.so libdnn_cublas.so

libdnn_openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^ -lopenblas

libdnn_avx.so: dnn_avx.c
	gcc -shared -fPIC -o $@ $^ -mavx2

libdnn_cublas.so: dnn_cublas.cu 
	nvcc -Xcompiler -fPIC -shared -o $@ $^ -lcublas

clean:
	rm -f *.so
