void GPU_initialize_Interface(int numOwnedPoints, GParam* param);
void divideNeighbor_Interface(int numOwnedPoints, GParam* param);
void Dkernel_Interface(GParam* param, int numOwnedpoints, double horizon, cudaStream_t stream0, int flag);
void Fkernel_Interface(GParam* param, int numOwnedpoints, double horizon, double bulkModulus, double shearModulus, cudaStream_t stream0, int flag);

__global__ void GPU_HelloWorld(int* a);
__global__ void GPU_ReverseWeightVolume(int numOwnedPoints, double* mp);
__global__ void GPU_initialize(int numOwnedPoints, double* x, double* v, double* a, double* u, double* y, double dt);
__global__ void divideNeighbor(int numOwnedPoints, int* neighborhoodlist, int* neighborPtr,int groupnum);
__global__ void DilatationKernel(double *x, double *y, double*m, double* c, double *d, int* bonddamage, int * neighborlist, int numOwnedpoints, double horizon, int* neighborPtr, int outflag);
__global__ void ForceKernel(double *x, double *y, double*m, double* c, double *d, int* bonddamage, double *f, int * neighborlist, int numOwnedpoints,double bulkModulus,double shearModulus, double horizon, int* neighborPtr, double* a, double* v, double* u, double density, double dt, double* yout,int outflag);