sweep: sweep.cu
	nvcc -G -g -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -I/usr/local/cuda/include -o sweep sweep.cu device_prop.cu
