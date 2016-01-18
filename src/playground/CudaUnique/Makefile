sweep: sweep.o device_prop.o
	nvcc -G -g -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -I/usr/local/cuda/include -o sweep sweep.o device_prop.o

sweep.o: sweep.cu
	nvcc -G -g -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -I/usr/local/cuda/include -c sweep.cu

device_prop.o: device_prop.cu
	nvcc -G -g -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -I/usr/local/cuda/include -c device_prop.cu

clean:
	rm -f sweep
	rm -f *.o
