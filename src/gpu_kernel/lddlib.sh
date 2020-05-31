nvcc -arch=sm_60 -std=c++11 -O3 -rdc=true -Xcompiler -fPIC -c gpukernel.cu -o gpukernel.o
mpicc  algorithm.o -fPIC -shared -o libgpukernel.so
