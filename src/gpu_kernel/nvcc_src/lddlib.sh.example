nvcc -arch=sm_60 -std=c++11 -O3  -Xcompiler -fPIC -c PD.cu -o PD.o
mpicc -fPIC -c gpukernel.cpp -o gpukernel.o
mpicc -O3 -fPIC -shared -L${CUDA_LIB}  -lcudart -lm PD.o gpukernel.o -o ../libgpukernel.so

