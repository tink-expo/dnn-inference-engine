all: libdnn_openblas.so libdnn_avx.so

libdnn_openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^ -lopenblas

libdnn_avx.so: dnn_avx.c
	gcc -shared -fPIC -o $@ $^ -mavx2

clean:
	rm -f libdnn_openblas.so libdnn_avx.so
