all: curve

curve.o: curve.cu
	nvcc -I/opt/local/include -c curve.cu

curve: curve.o
	nvcc -o curve curve.o -L/opt/local/lib -lnetpbm

