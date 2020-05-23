libdnnrun.so: dnn_run.c
	gcc -shared -fPIC -o $@ $^ -lopenblas

clean:
	rm -f libdnnrun.so
