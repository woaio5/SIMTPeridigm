hipcc -c -O3 -std=c++11 -fPIC -D__HIP_PLATFORM_HCC__ -o gpukernel.o gpukernel.cpp
hipcc -fPIC -shared -std=c++11 gpukernel.o -o ../libgpukernel.so

