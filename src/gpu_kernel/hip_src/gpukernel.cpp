#include "hip/hip_runtime.h"
#include <iostream>
#include "gpukernel.h"
#include "mpi.h"
#include <math.h>
#define BLOCKSIZE 256 

GParam *gparamlist;
CParam *cparamlist;
int Initflag;
int nowblock;
int myrank;
int* newNptr;
int* newPtr;
int* mybondDamage;
#define groupsize 4
double tot,dtot,ftot,dmpi,fmpi,dcpy,fcpy,dken,fken;
hipStream_t stream0, stream1;

double mysecond(){
        struct timeval tp;
        struct timezone tpz;
        int i = gettimeofday(&tp, &tpz);
        return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void GPU_HelloWorld(int* a){
	int i;
	i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	a[i] = i;
}

__global__ void GPU_ReverseWeightVolume(int numOwnedPoints, double* mp){
	int i,j,k;
	i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	if(i<numOwnedPoints){
		mp[i] = 1.0/mp[i];
	}
}

__global__ void GPU_initialize(int numOwnedPoints, double* x, double* v, double* a, double* u, double* y, double dt){
	int i;
	i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	if(i<numOwnedPoints){
		v[i*3] = (200-50*pow(((x[i*3+2]/0.05)-1),2))*cos(atan2(x[i*3+1],x[i*3]));
                v[i*3+1] = (200-50*pow(((x[i*3+2]/0.05)-1),2))*sin(atan2(x[i*3+1],x[i*3]));
                v[i*3+2] = 100*(x[i*3+2]/0.05-1);
                y[i*3] = y[i*3] + dt * v[i*3];
                y[i*3+1] = y[i*3+1] + dt * v[i*3+1];
                y[i*3+2] = y[i*3+2] + dt * v[i*3+2];
                u[i*3] = dt * v[i*3];
                u[i*3+1] = dt * v[i*3+1];
                u[i*3+2] = dt * v[i*3+2];

	}
}

__global__ void divideNeighbor(int numOwnedPoints, int* neighborhoodlist, int* neighborPtr,int groupnum){
	int i;
	i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	if(i<groupnum){
		int *Nptr = &neighborhoodlist[neighborPtr[i]];
		int numNeigh = *Nptr;
		Nptr++;
		int j;
		for(j=0;j<numNeigh;j++){
			if(Nptr[j]>=numOwnedPoints)
				break;
		}
		
		neighborhoodlist[neighborPtr[i]] |= (j<<16);
	}
}


void InitVUY(GParam* param, int numOwnedPoints){
	int i;
	int nthreads = numOwnedPoints+param->TotalImport;
	int nblocks = (numOwnedPoints+param->TotalImport)/BLOCKSIZE+1;
	hipLaunchKernelGGL(GPU_initialize, dim3(nblocks),dim3(BLOCKSIZE),0,0,numOwnedPoints, param->x, param->v, param->a, param->u, param->y, param->dt);
	hipDeviceSynchronize();
}


void InitDeviceMemory(GParam* param,int numOwnedPoints, int neighborsize, double*x , double *y,double* weightvolume, double* cellvolume, int* bonddamage, int* neighborhoodlist, int* neighborPtr,int globalnum){
	int i;
	hipStreamCreate(&stream0);
	hipStreamCreate(&stream1);
	hipMalloc((void**)&(param->x),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	hipMalloc((void**)&(param->y),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	hipMalloc((void**)&(param->v),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	hipMalloc((void**)&(param->a),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	hipMalloc((void**)&(param->u),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	hipMalloc((void**)&(param->force),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	hipMalloc((void**)&(param->weightvolume),(numOwnedPoints+param->TotalImport)*sizeof(double));
	hipMalloc((void**)&(param->yout), (numOwnedPoints+param->TotalImport)*3*sizeof(double));
	//hipMalloc((void**)&(param->density),globalnum/3*sizeof(double));
	hipMalloc((void**)&(param->cellvolume),(numOwnedPoints+param->TotalImport)*sizeof(double));
	hipMalloc((void**)&(param->dilatation),(numOwnedPoints+param->TotalImport)*sizeof(double));
	hipMalloc((void**)&(param->neighborPtr),numOwnedPoints*sizeof(int));
	hipMalloc((void**)&(param->mybondDamage),neighborsize*sizeof(int));
	hipMalloc((void**)&(param->neiborlist),neighborsize*sizeof(int));
	hipMemcpy(param->x, x, (numOwnedPoints+param->TotalImport)*3*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(param->y, y, (numOwnedPoints+param->TotalImport)*3*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(param->cellvolume, cellvolume, (numOwnedPoints+param->TotalImport)*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(param->weightvolume, weightvolume, (numOwnedPoints+param->TotalImport)*sizeof(double),hipMemcpyHostToDevice);
	//hipMemcpy(param->density, density, globalnum/3*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpy(param->mybondDamage, bonddamage, (neighborsize)*sizeof(int),hipMemcpyHostToDevice);
	hipMemcpy(param->neiborlist, neighborhoodlist, neighborsize*sizeof(int),hipMemcpyHostToDevice);
	hipMemcpy(param->neighborPtr, neighborPtr, (numOwnedPoints/4+(numOwnedPoints%groupsize!=0))*sizeof(int),hipMemcpyHostToDevice);
	//newNptr = (int *)malloc(sizeof(int)*neighborsize);
	//newPtr = (int *)malloc(sizeof(int)*numOwnedPoints/4+(numOwnedPoints%groupsize!=0));
	mybondDamage = (int *)malloc(sizeof(int)*(neighborsize));
	int deviceID = myrank % 4;
        hipGetDevice(&deviceID);
	int nthreads = numOwnedPoints+param->TotalImport;
	int nblocks = (numOwnedPoints+param->TotalImport)/BLOCKSIZE+1;
	//int globalnum = param->numOwnedpoints;
	printf("Initial Memory on Device %e %e %d\n",param->dt, x[0],param->TotalImport);
	//hipLaunchKernelGGL(GPU_initialize, dim3(nblocks),dim3(BLOCKSIZE),0,0,numOwnedPoints, param->x, param->v, param->a, param->u, param->y, param->dt);
	//hipDeviceSynchronize();
	hipLaunchKernelGGL(divideNeighbor, dim3(nblocks),dim3(BLOCKSIZE),0,0,numOwnedPoints, param->neiborlist, param->neighborPtr, numOwnedPoints/groupsize+(numOwnedPoints%groupsize!=0));
	hipDeviceSynchronize();
	hipLaunchKernelGGL(GPU_ReverseWeightVolume, dim3(nblocks),dim3(BLOCKSIZE),0,0,numOwnedPoints+param->TotalImport, param->weightvolume);
	hipDeviceSynchronize();
	//hipMemcpy(newNptr, param->neiborlist,neighborsize*sizeof(int),hipMemcpyDeviceToHost);
	//hipMemcpy(newPtr, param->neighborPtr, (numOwnedPoints/4+(numOwnedPoints%groupsize!=0))*sizeof(int),hipMemcpyDeviceToHost);
	int st = neighborPtr[0];
	//for(int i=0; i< neighborhoodlist[st];i++)
		printf("inner:%d outter:%d\n",(neighborhoodlist[0]&0x0000ffff),(neighborhoodlist[0]&0xffff0000)>>16);
	//printf("\n");
	
}

void CPU_HelloWorld(int *a){
	int i;
	int *deva;
	//hipMalloc((void **)&deva, 1024*sizeof(int));
	//printf("~~~~CPU HELLOWOLRD!!!\n");
	for(i=0;i<1024;i++)
		a[i] = 0;
	/*hipMemcpy(deva, a, 1024*sizeof(int), hipMemcpyHostToDevice);
	hipLaunchKernelGGL(GPU_HelloWorld, dim3(8),dim3(128),0,0,deva);
	hipMemcpy(a, deva, 1024*sizeof(int), hipMemcpyDeviceToHost);
	if(a[128]==128)
		printf("~~~GPU is started successfully!!!!!\n");*/
}

__global__ void DilatationKernel(double *x, double *y, double*m, double* c, double *d, int* bonddamage, int * neighborlist, int numOwnedpoints, double horizon, int* neighborPtr, int outflag){
	int i,j,k;
	i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	numOwnedpoints = numOwnedpoints + (groupsize - numOwnedpoints%groupsize)*(numOwnedpoints%groupsize!=0);
	if(i< numOwnedpoints){
		int *Nptr = &neighborlist[neighborPtr[i/groupsize]];
		int st,ed;
		int numNeigh = ((*Nptr)&0x0000ffff)-outflag*(((*Nptr)&0xffff0000)>>16);
		int *bond = &bonddamage[neighborPtr[i/groupsize]]; 
		int mid = ((*Nptr)&0xffff0000)>>16;
		int total = ((*Nptr)&0x0000ffff);
		st = outflag*mid;
		ed = mid + outflag*(total - mid);
		Nptr++;
		Nptr += st;
		bond += st;
		numNeigh = ed-st;
		double ixx =x[i*3];	
		double ixy =x[i*3+1];	
		double ixz =x[i*3+2];	
		double iyx =y[i*3];	
		double iyy =y[i*3+1];	
		double iyz =y[i*3+2];
		double di = d[i]*(outflag);
		double xdx,xdy,xdz,initdist,ydx,ydy,ydz,currdist,e;
		double s;
		int ci = i/groupsize;
		int position =  i % groupsize;
		int loop = numNeigh/groupsize /*+ (numNeigh%groupsize != 0)*/;
		double rx[4],ry[4];
		int rc[4];
		int stid = (i/groupsize)*groupsize;
		int damage;
		for(j=0;j<loop;j++){
			int p = Nptr[j*groupsize + position];
			rx[0] = x[p*3];
			rx[1] = x[p*3+1];
			rx[2] = x[p*3+2];
			rx[3] = c[p]; 
			ry[0] = y[p*3];
			ry[1] = y[p*3+1];
			ry[2] = y[p*3+2];
			ry[3] = m[p];
			for(k=0;k<groupsize;k++){
				//int thisp = __shlf(p, stid+(position+k)%groupsize);
				damage = (bond[j*groupsize+(position+k)%groupsize]>>position)&0x00000001;
				double xjx = __shfl(rx[0], stid+(position+k)%groupsize, 64);
				double xjy = __shfl(rx[1], stid+(position+k)%groupsize, 64);
				double xjz = __shfl(rx[2], stid+(position+k)%groupsize, 64);
				double cj = __shfl(rx[3], stid+(position+k)%groupsize, 64);
				double yjx = __shfl(ry[0], stid+(position+k)%groupsize, 64);
				double yjy = __shfl(ry[1], stid+(position+k)%groupsize, 64);
				double yjz = __shfl(ry[2], stid+(position+k)%groupsize, 64);
				double mj = __shfl(ry[3], stid+(position+k)%groupsize, 64);
				xdx = ixx - xjx;
				xdy = ixy - xjy;
				xdz = ixz - xjz;
                        	initdist = xdx*xdx + xdy*xdy + xdz*xdz;
				int flag = (initdist<=horizon*horizon);
				ydx = yjx - iyx;
				ydy = yjy - iyy;
				ydz = yjz - iyz;
                        	currdist = ydx*ydx + ydy*ydy + ydz*ydz;
				double currinit = sqrt(initdist*currdist);
                        	//e = currdist - initdist;
				s = (currdist- 2*currinit + initdist);
				int bflag = (s>0.02*0.02*initdist)*((currdist-initdist)>0);
				damage = bflag | damage;
				di += flag*3*(1-damage)*(currinit - initdist)*cj*m[i];
				rc[(k+position)%groupsize] = damage<<position;
				//atomicOr(&bond[j*groupsize+(k+position)%groupsize],damage<<position);
				
			}
			int dam = 0;
			dam |= rc[position];
			for(k=1;k<groupsize;k++){
				dam |= __shfl(rc[(position+k)%groupsize], stid+(position-k+groupsize)%groupsize, 64);
			}
			bond[j*groupsize+position] = dam;
			
		
		}
		//rest = numNeigh % groupsize;
		for(int n = j*groupsize; n<numNeigh; n++){
			int p = Nptr[n];
			xdx = ixx - x[p*3];	
			xdy = ixy - x[p*3+1];	
			xdz = ixz - x[p*3+2];
			int damage = (bond[n] >> position)&0x00000001;
			//int damage = 0;
                        initdist = xdx*xdx + xdy*xdy + xdz*xdz;
			int flag = (initdist<=horizon*horizon);
                        ydx = y[p*3] - iyx;
                        ydy = y[p*3+1] - iyy;
                        ydz = y[p*3+2] - iyz;
                        currdist = ydx*ydx + ydy*ydy + ydz*ydz;
			double currinit = sqrt(initdist*currdist);
                        //e = currdist - initdist;
			s = (currdist - 2*currinit + initdist);
			int bflag = (s>0.02*0.02*initdist)*((currdist-initdist)>0);
			damage = bflag | damage;
			//if(s > 0.02)
			//	*bond = 1.0;
			di += flag*3*(1-damage)*(currinit - initdist )*c[p]*m[i];
			atomicOr(&bond[n],damage<<position);
		}
	     d[i] = di;	
		
	}
}


__global__ void ForceKernel(double *x, double *y, double*m, double* c, double *d, int* bonddamage, double *f, int * neighborlist, int numOwnedpoints,double bulkModulus,double shearModulus, double horizon, int* neighborPtr, double* a, double* v, double* u, double density, double dt, double* yout,int outflag){
	int i,j,k;
	i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	numOwnedpoints = numOwnedpoints + (groupsize - numOwnedpoints%groupsize)*(numOwnedpoints%groupsize!=0);
	if(i< numOwnedpoints){
		int *Nptr = &neighborlist[neighborPtr[i/groupsize]];
		int st,ed;
		int numNeigh = ((*Nptr)&0x0000ffff)-outflag*(((*Nptr)&0xffff0000)>>16);
		int *bond = &bonddamage[neighborPtr[i/groupsize]]; 
		int mid = ((*Nptr)&0xffff0000)>>16;
		int total = ((*Nptr)&0x0000ffff);
		st = outflag*mid;
		ed = mid + outflag*(total - mid);
		Nptr++;
		Nptr += st;
		bond += st;
		numNeigh = ed-st;
		/*int *Nptr = &neighborlist[neighborPtr[i/groupsize]];
		int *bond = &bonddamage[neighborPtr[i/groupsize]]; 
		int numNeigh = ((*Nptr)&0x0000ffff);
		Nptr++;*/
		double K = bulkModulus;
		double MU = shearModulus;
		double ixx =x[i*3];	
		double ixy =x[i*3+1];	
		double ixz =x[i*3+2];	
		double iyx =y[i*3];	
		double iyy =y[i*3+1];	
		double iyz =y[i*3+2];
		double fxi = f[i*3]*outflag;
		double fyi = f[i*3+1]*outflag;
		double fzi = f[i*3+2]*outflag;
		double xdx,xdy,xdz,initdist,ydx,ydy,ydz,currdist,e;
		int position =  i % groupsize;
		int ci = i/groupsize;
		//int position =  i % groupsize;
		int loop = numNeigh/groupsize /*+ (numNeigh%groupsize != 0)*/;
		double rx[4],ry[4],rd;
		int rc[4];
		int stid = (i/groupsize)*groupsize;
		int damage;
		long i1;
		long magic = 0x5fe6ec85e7de30da;
		double x2,q,r,z;
		double thr = 1.0/3.0;
		for(j=0;j<loop;j++){
			int p = Nptr[j*groupsize + position];
			rx[0] = x[p*3];
			rx[1] = x[p*3+1];
			rx[2] = x[p*3+2];
			rx[3] = c[p]; 
			ry[0] = y[p*3];
			ry[1] = y[p*3+1];
			ry[2] = y[p*3+2];
			ry[3] = m[p];
			rd = d[p];
			for(k=0;k<groupsize;k++){
				damage = (bond[j*groupsize+(k+position)%groupsize]>>position)&0x00000001;
				double xjx = __shfl(rx[0], stid+(position+k)%groupsize, 64);
				double xjy = __shfl(rx[1], stid+(position+k)%groupsize, 64);
				double xjz = __shfl(rx[2], stid+(position+k)%groupsize, 64);
				double cj = __shfl(rx[3], stid+(position+k)%groupsize, 64);
				double yjx = __shfl(ry[0], stid+(position+k)%groupsize, 64);
				double yjy = __shfl(ry[1], stid+(position+k)%groupsize, 64);
				double yjz = __shfl(ry[2], stid+(position+k)%groupsize, 64);
				double mj = __shfl(ry[3], stid+(position+k)%groupsize, 64);
				double dj = __shfl(rd, stid+(position+k)%groupsize, 64);
				xdx = ixx - xjx;
				xdy = ixy - xjy;
				xdz = ixz - xjz;
				double initdist2 = xdx*xdx + xdy*xdy + xdz*xdz;
                        	initdist = sqrt(xdx*xdx + xdy*xdy + xdz*xdz);
				int flag = (initdist<=horizon);
                        	ydx = yjx - iyx;
                        	ydy = yjy - iyy;
                        	ydz = yjz - iyz;
                        	currdist = ydx*ydx + ydy*ydy + ydz*ydz;
				x2 = currdist * 0.5;
				i1 = *(long *)&currdist;
				i1 = magic - (i1>>1);
				z = *(double *)&i1;
				z = z*(1.5-x2*z*z);
				z = z*(1.5-x2*z*z);
				z = z*(1.5-x2*z*z);
				z = z*(1.5-x2*z*z);
				z = z*(1.5-x2*z*z);
				double temp = z;
                        	e = (1.0 - initdist*temp)*(currdist != initdist2);
                        	//e = 1.0 - initdist*temp;
                        	double zeroflag = (double)(currdist == 0);
                        	double alpha = 15.0*MU*m[i];
                        	double alphap = 15.0*MU*mj;
                        	double c1 = 1 * d[i] * (3.0*K*m[i] - alpha*thr);
                        	double cp = 1 * dj * (3.0*K*mj - alphap*thr);
                        	double t = (1-damage) * (c1* initdist*temp + (1-damage)*1*alpha*e)*flag;
                        	double tp = (1-damage) * (cp* initdist*temp + (1-damage)*1*alphap*e)*flag;
                        	double fx = t * ydx;
                        	double fy = t * ydy;
                        	double fz = t * ydz;
                        	double fxp = tp * ydx;
                        	double fyp = tp * ydy;
                        	double fzp = tp * ydz;
                        	fxi += fx *cj;
                        	fyi += fy *cj;
                        	fzi += fz *cj;
                        	fxi += fxp *cj;
                        	fyi += fyp *cj;
                        	fzi += fzp *cj;
			}
		}
		for(int n = j*groupsize; n<numNeigh; n++){
			int p = Nptr[n];
                        xdx = ixx - x[p*3];
                        xdy = ixy  - x[p*3+1];
                        xdz = ixz - x[p*3+2];
			double initdist2 = xdx*xdx + xdy*xdy + xdz*xdz;
                        initdist = sqrt(xdx*xdx + xdy*xdy + xdz*xdz);
			damage = (bond[n] >> position)&0x00000001;
			int flag = (initdist<=horizon);
                        //flag = (double)(initdist  <= horizon);
                        ydx = y[p*3] - iyx;
                        ydy = y[p*3+1] - iyy;
                        ydz = y[p*3+2] - iyz;
                        currdist = ydx*ydx + ydy*ydy + ydz*ydz;
                        double zeroflag = (double)(currdist == 0);
			x2 = currdist * 0.5;
			i1 = *(long *)&currdist;
			i1 = magic - (i1>>1);
			z = *(double *)&i1;
			z = z*(1.5-x2*z*z);
			z = z*(1.5-x2*z*z);
			z = z*(1.5-x2*z*z);
			z = z*(1.5-x2*z*z);
			z = z*(1.5-x2*z*z);
			double temp = z;
                        e = (1.0 - initdist*temp)*(currdist != initdist2);
                      	//zeroflag = 1.0e-16;
                        double alpha = 15.0*MU*m[i];
                        double alphap = 15.0*MU*m[p];
                        double c1 = 1 * d[i] * (3.0*K*m[i] - alpha*thr);
                        double cp = 1 * d[p] * (3.0*K*m[p] - alphap*thr);
                        double t = (1-damage) * (c1* initdist*temp + (1-damage)*1*alpha*e)*flag;
                        double tp = (1-damage) * (cp* initdist*temp + (1-damage)*1*alphap*e)*flag;
                        double fx = t * ydx;
                        double fy = t * ydy;
                        double fz = t * ydz;
                        double fxp = tp * ydx;
                        double fyp = tp * ydy;  
                        double fzp = tp * ydz;
                        fxi += fx *c[p];
                        fyi += fy *c[p];
                        fzi += fz *c[p];
                        fxi += fxp *c[p];
                        fyi += fyp *c[p];
                        fzi += fzp *c[p];
		}
		f[i*3] = fxi;
		f[i*3+1] = fyi;
		f[i*3+2] = fzi;
		double a1,a2,a3,v1,v2,v3;	
		a1 = fxi/density;
		a2 = fyi/density;
		a3 = fzi/density;
		a[i*3] = a1; 
		a[i*3+1] = a2; 
		a[i*3+2] = a3;
		v1 = v[i*3]+(a1*dt*0.5+a1*dt*0.5)*outflag;
		v2 = v[i*3+1]+(a2*dt*0.5+ a2*dt*0.5)*outflag;
		v3 = v[i*3+2]+(a3*dt*0.5 + a3*dt*0.5)*outflag;

		//v2 += a2*dt*0.5;
		//v3 += a2*dt*0.5;
		v[i*3] = v1;
		v[i*3+1]= v2;
		v[i*3+2] = v3;
		u[i*3] += dt*v1*outflag;		
		u[i*3+1] += dt*v2*outflag;		
		u[i*3+2] += dt*v3*outflag;
		yout[i*3] = iyx+dt*v1*outflag;		
		yout[i*3+1] = iyy+dt*v2*outflag;		
		yout[i*3+2] = iyz+dt*v3*outflag;
				
	}
	
}

void ForceTest(double *x, double *y, double*m, double* c, double *d, double* bonddamage, double *f, const int * neighborlist, int numOwnedpoints,double bulkModulus,double shearModulus, double horizon, int* neighborPtr){
	int i = 0;
	//i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	if(i< numOwnedpoints){
		const int *Nptr = &neighborlist[neighborPtr[i/4]];
		double *bond = &bonddamage[neighborPtr[i/4]-i/4]; 
		int numNeigh = *Nptr;
		Nptr++;
		double K = bulkModulus;
		double MU = shearModulus;
		double ixx =x[i*3];	
		double ixy =x[i*3+1];	
		double ixz =x[i*3+2];	
		double iyx =y[i*3];	
		double iyy =y[i*3+1];	
		double iyz =y[i*3+2];
		double fxi = 0;
		double fyi = 0;
		double fzi = 0;
		double xdx,xdy,xdz,initdist,ydx,ydy,ydz,currdist,e;
		for(int n =0; n<numNeigh; n++,Nptr++, bond++){
			int p = *Nptr;
                        xdx = ixx - x[p*3];
                        xdy = ixy  - x[p*3+1];
                        xdz = ixz - x[p*3+2];
                        initdist = sqrt(xdx*xdx + xdy*xdy + xdz*xdz);
                        int flag = (double)(initdist  <= horizon);
                        ydx = y[p*3] - iyx;
                        ydy = y[p*3+1] - iyy;
                        ydz = y[p*3+2] - iyz;
                        currdist = sqrt(ydx*ydx + ydy*ydy + ydz*ydz);
                        e = currdist - initdist;
                        double zeroflag = (double)(currdist == 0);
                      	//zeroflag = 1.0e-16;
                        double alpha = 15.0*MU/m[i];
                        double alphap = 15.0*MU/m[p];
                        double c1 = 1 * d[i] * (3.0*K/m[i] - alpha/3.0);
                        double cp = 1 * d[p] * (3.0*K/m[p] - alphap/3.0);
                        double t = (1-0) * (c1* initdist + (1-0)*1*alpha*e)*flag;
                        double tp = (1-0) * (cp* initdist + (1-0)*1*alphap*e)*flag;
                        double fx = t * ydx/(currdist+zeroflag);
                        double fy = t * ydy/(currdist+zeroflag);
                        double fz = t * ydz/(currdist+zeroflag);
                        double fxp = tp * ydx/(currdist+zeroflag);
                        double fyp = tp * ydy/(currdist+zeroflag);
                        double fzp = tp * ydz/(currdist+zeroflag);
			//if(myrank == 1)
			//	printf("%d MyForce:%e\n",p, fxp*c[p]);
                        fxi += fx *c[p];
                        fyi += fy *c[p];
                        fzi += fz *c[p];
                        fxi += fxp *c[p];
                        fyi += fyp *c[p];
                        fzi += fzp *c[p];
		}
		f[i*3] = fxi;
		f[i*3+1] = fyi;
		f[i*3+2] = fzi;
		//a[i*3] = fxi/density;
	}
	printf("%d: Test Force:%e\n", myrank, f[0]);
	
}

void GPU_Dilatation_Interface(double *x, double *y, double *weightvolume, double* cellvolume,double *dilatation, double* bondDamage, const int* neighborhoodList, int numOwnedPoints, double horizon, double m_alpha, double* deltaTemperature, GParam* param ){
	//int i;
	int i;
	double st,ed;
	double* hostd;
	int nthreads = numOwnedPoints;
	int nblocks = numOwnedPoints/BLOCKSIZE+1;
	int globalnum = param->numOwnedpoints;
	int mpi_size;
	MPI_Status* mpistat;
	MPI_Request* mpireq;
  	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  	mpistat = (MPI_Status *)malloc(sizeof(MPI_Status)*mpi_size*2);
  	mpireq = (MPI_Request *)malloc(sizeof(MPI_Request)*mpi_size*4);
	dmpi=0;dcpy=0;dken=0;

	st = mysecond();
	//param->bulkModulus = bulkModulus;
	//param->shearModulus = shearModulus;
	//param->horizon = horizon;
	
	double ***sendbuf = param->sendbuf;
	double ***recvbuf = param->recvbuf;
	int **NumImport = param->NumImport;
	int **NumProcRequest = param->NumProcRequest;
	int ***ProcRequestList = param->ProcRequestList;
	int ***ProcExportList = param->ProcExportList;
	int **LIDList = param->LIDList;
	int MaxGID = param->MaxGID;
	int MinGID = param->MinGID;

	//double* temp = (double *)malloc(sizeof(double)*globalnum);
	hipLaunchKernelGGL(DilatationKernel, dim3(nblocks),dim3(BLOCKSIZE),0,stream0,param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->neiborlist, numOwnedPoints, horizon, param->neighborPtr,0);
	hipMemcpyAsync(y, param->y, numOwnedPoints*3*sizeof(double), hipMemcpyDeviceToHost, stream1);
	hipStreamSynchronize(stream1);
	for(i=0;i<mpi_size;i++){
		if(i!=myrank && NumProcRequest[nowblock][i]){
			for(int j=0;j<NumProcRequest[nowblock][i];j++){
				int nid = ProcExportList[nowblock][i][j];
				int lid = LIDList[nowblock][nid-MinGID];
				sendbuf[nowblock][i][j*3] = y[lid*3];
				sendbuf[nowblock][i][j*3+1] = y[lid*3+1];
				sendbuf[nowblock][i][j*3+2] = y[lid*3+2];
			}
			MPI_Isend(sendbuf[nowblock][i], NumProcRequest[nowblock][i]*3, MPI_DOUBLE, i, myrank*10+i, MPI_COMM_WORLD, &mpireq[i*2] );
			MPI_Irecv(recvbuf[nowblock][i], NumImport[nowblock][i]*3, MPI_DOUBLE, i, i*10+myrank, MPI_COMM_WORLD, &mpireq[i*2+1]);
			
		}
	}
	for(i=0;i<mpi_size;i++){
		if(i!=myrank && NumProcRequest[nowblock][i]!=0){
                        MPI_Wait(&mpireq[i*2],&mpistat[i*2]);
                        MPI_Wait(&mpireq[i*2+1],&mpistat[i*2+1]);
		}
	}
	for(i=0;i<mpi_size;i++){
		for(int j=0;j<NumImport[nowblock][i];j++){
			int nid = ProcRequestList[nowblock][i][j];	
			int lid = LIDList[nowblock][nid-MinGID];
			y[lid*3] = recvbuf[nowblock][i][j*3];
			y[lid*3+1] = recvbuf[nowblock][i][j*3+1];
			y[lid*3+2] = recvbuf[nowblock][i][j*3+2];
			
		}
	}
	/*hipHostMalloc((void**)&hostd, sizeof(double)*globalnum);
	st = mysecond();
	memcpy(hostd, y, globalnum*sizeof(double));
	hipMemcpy(param->y,hostd,globalnum*sizeof(double),hipMemcpyHostToDevice);
	ed = mysecond();
	printf("DIlatation: Memcpy Pined Y %e\n",ed-st);*/
	//int **NumImport = param->NumImport;
	int TotalImport = 0;
	for(i=0;i<mpi_size;i++){
		TotalImport += NumImport[nowblock][i];
	}
	//hipMemcpy(param->y,y,globalnum*sizeof(double),hipMemcpyHostToDevice);
	//printf("DIlatation: Memcpy Y %e, %d\n",ed-st, numOwnedPoints);
	//hipMemcpy(param->bonddamage, param->mybondDamage, (param->neighborsize-numOwnedPoints/groupsize)*sizeof(double),hipMemcpyHostToDevice);
	hipMemcpyAsync(&(param->y[numOwnedPoints*3]),&y[numOwnedPoints*3],(TotalImport)*3*sizeof(double),hipMemcpyHostToDevice,stream1);
	hipStreamSynchronize(stream1);
	//ed = mysecond();
	//dmpi += ed - st;
	//hipDeviceSynchronize();
	//st = mysecond();
	hipStreamSynchronize(stream0);
	//hipMemcpy(&(param->y[numOwnedPoints*3]),&y[numOwnedPoints*3],(TotalImport)*3*sizeof(double),hipMemcpyHostToDevice);
	//st = mysecond();
	hipLaunchKernelGGL(DilatationKernel, dim3(nblocks),dim3(BLOCKSIZE),0,stream0,param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->neiborlist, numOwnedPoints, horizon, param->neighborPtr,1);
	hipStreamSynchronize(stream0);
	//ahipDeviceSynchronize();
	//ed = mysecond();
	//dmpi += ed - st;
	//printf("DIlatation: Kernel %e\n",ed-st);
	/*hipMemcpy( mybondDamage ,param->mybondDamage, (param->neighborsize)*sizeof(int),hipMemcpyDeviceToHost);
	int *Nptr = &newNptr[newPtr[14823/4]];
	int numneigh = *Nptr;
	Nptr++;
	int *bond = &mybondDamage[newPtr[14823/4]];
	int position = 14823%4;
	for(int i = 0; i< numneigh; i++){
		int p = Nptr[i];
		double xdx = x[14823*3] - x[p*3];
		double xdy = x[14823*3+1] - x[p*3+1];
		double xdz = x[14823*3+2] - x[p*3+2];
		double dist = sqrt(xdx*xdx + xdy*xdy + xdz*xdz);
		double ydx = y[14823*3] - y[p*3];
		double ydy = y[14823*3+1] - y[p*3+1];
		double ydz = y[14823*3+2] - y[p*3+2];
		double cdist = sqrt(ydx*ydx + ydy*ydy + ydz*ydz);
		
		double flag = (dist<= horizon);
		if(((bond[i]>>position)&0x00000001)==1 && flag == 1)
			printf("(%d,%e), ",Nptr[i],(cdist-dist)/dist);
	}
	printf("\n");*/
	//st = mysecond();
	//hipMemcpy(&dilatation[numOwnedPoints], &((param->dilatation)[numOwnedPoints]), TotalImport*sizeof(double),hipMemcpyDeviceToHost);
	//hipMemcpy(dilatation, param->dilatation, numOwnedPoints*sizeof(double),hipMemcpyDeviceToHost);
	//ed = mysecond();
	//dcpy += ed -st;
	//hipStreamDestroy(stream0);
	//hipStreamDestroy(stream1);
	ed = mysecond();
	dcpy += ed - st;
	/*char result[1000];
	sprintf(result,"+++++++++++++++++++++++++\nDilatation Kernel BreakDown of %d\nMPI:%e\nMemcpy:%e\nKernel:%e\n+++++++++++++++++++++++++\n",myrank,mpi,cpy,ken);
	printf("%s",result);*/
	//printf("DIlatation: Memcpy dilatation %e\n",ed-st);
	//printf("DIlatation: Memcpy dilatation %e\n",ed-st);
}


void GPU_Force_Interface(double *x, double *y, double* weightvolume, double* cellvolume, double* dilatation, double* bondDamage, double* force, double* partialStress, const int* neighborhoodList, int numOwnedPoints, double bulkModulus, double shearModulus, double horizon, double alpha, double* deltaTemperature, GParam* param, double* oriforce ){
	int i;
	int nthreads = numOwnedPoints;
	int nblocks = numOwnedPoints/BLOCKSIZE+1;
	int globalnum = param->numOwnedpoints;
  	int mpi_size;
	double st, ed;
	fmpi=0;
	fcpy=0;
	fken=0;
	MPI_Status* mpistat;
	MPI_Request* mpireq;
  	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  	mpistat = (MPI_Status *)malloc(sizeof(MPI_Status)*mpi_size*2);
  	mpireq = (MPI_Request *)malloc(sizeof(MPI_Request)*mpi_size*4);
	param->bulkModulus = bulkModulus;
	param->shearModulus = shearModulus;
	param->horizon = horizon;
	double ***sendbuf = param->sendbuf;
	double ***recvbuf = param->recvbuf;
	int **NumImport = param->NumImport;
	int **NumProcRequest = param->NumProcRequest;
	int ***ProcRequestList = param->ProcRequestList;
	int ***ProcExportList = param->ProcExportList;
	int **LIDList = param->LIDList;
	int MaxGID = param->MaxGID;
	int MinGID = param->MinGID;
	st = mysecond();
	//hipStream_t stream0, stream1;
	//hipStreamCreate(&stream0);
	//hipStreamCreate(&stream1);
	int TotalImport = 0;
	for(i=0;i<mpi_size;i++){
		TotalImport += NumImport[nowblock][i];	
	}
	//printf("Before myrank %d: %d in dilatation[%d] = %e\n", myrank ,35250, LIDList[nowblock][35250-MinGID], dilatation[35250]);
	//st = mysecond();
	hipLaunchKernelGGL(ForceKernel, dim3(nblocks),dim3(BLOCKSIZE),0,stream0,param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->force, param->neiborlist, numOwnedPoints, bulkModulus,shearModulus,horizon,param->neighborPtr, param->a, param->v, param->u, param->density,param->dt,param->yout,0);
	hipMemcpyAsync(dilatation, param->dilatation, numOwnedPoints*sizeof(double), hipMemcpyDeviceToHost, stream1);
	hipStreamSynchronize(stream1);
	for(i=0;i<mpi_size;i++){
		if(i!=myrank && NumProcRequest[nowblock][i]){
			for(int j=0;j<NumProcRequest[nowblock][i];j++){
				int nid = ProcExportList[nowblock][i][j];
				int lid = LIDList[nowblock][nid-MinGID];
				sendbuf[nowblock][i][j] = dilatation[lid];
			}
			MPI_Isend(sendbuf[nowblock][i], NumProcRequest[nowblock][i], MPI_DOUBLE, i, myrank*10+i, MPI_COMM_WORLD, &mpireq[i*2] );
			MPI_Irecv(recvbuf[nowblock][i], NumImport[nowblock][i], MPI_DOUBLE, i, i*10+myrank, MPI_COMM_WORLD, &mpireq[i*2+1]);
			
		}
	}
	for(i=0;i<mpi_size;i++){
		if(i!=myrank && NumProcRequest[nowblock][i]!=0){
                        MPI_Wait(&mpireq[i*2],&mpistat[i*2]);
                        MPI_Wait(&mpireq[i*2+1],&mpistat[i*2+1]);
		}
	}
	for(i=0;i<mpi_size;i++){
		for(int j=0;j<NumImport[nowblock][i];j++){
			int nid = ProcRequestList[nowblock][i][j];	
			int lid = LIDList[nowblock][nid-MinGID];
			dilatation[lid] = recvbuf[nowblock][i][j];
			
		}
	}
	//printf("Force: MPI %e, %d\n",ed-st, TotalImport);
	//st = mysecond();
	//hipMemcpy(param->y,y,globalnum*sizeof(double),hipMemcpyHostToDevice);
	//hipMemcpy(param->weightvolume,weightvolume,globalnum/3*sizeof(double),hipMemcpyHostToDevice);
	//hipMemcpy(param->dilatation,dilatation,globalnum/3*sizeof(double),hipMemcpyHostToDevice);
	//ed = mysecond();
	//printf("Force: Memcpy Y and D %e\n",ed-st);
	int *neighborPtr = (int *)malloc(sizeof(int)*numOwnedPoints);
	/*hipMemcpy(force, param->force, (numOwnedPoints)*3*sizeof(double),hipMemcpyDeviceToHost);
  	for( i=0;i<numOwnedPoints*3;i++){
	//if(!isfinite(force1[i]))
	//	printf(\n");
		if(fabs(force[i]- oriforce[i])/fabs(oriforce[i])>1.0e-7 || !isfinite(oriforce[i])){
			printf("%d : Force Wrong! %d: %e, %e\n",myrank, i, force[i], oriforce[i]);
			//cout<<myrank<<" : Force Wrong! "<<i<<":"<<force[i]<<","<<oriforce[i]<<endl;
			break;
		}
 	 }
  	if(i == numOwnedPoints*3)
		printf("%d : Force Right\n",myrank);*/
		//cout<<myrank<<" : Force Right!"<<endl;
	//hipMemcpy(neighborPtr, param->neighborPtr, (numOwnedPoints)*sizeof(int ),hipMemcpyDeviceToHost);
	//printf("Thanks for using the GPU version!!!! %p,%d\n",param->force,globalnum);
	//printf("After myrank %d: %d in dilatation[%d] = %e\n", myrank ,35250, LIDList[nowblock][35250-MinGID], weightvolume[35250]);
	//param->numOwnedpoints = numOwnedPoints;
	//st = mysecond();
	hipMemcpyAsync(&(param->dilatation[numOwnedPoints]),&dilatation[numOwnedPoints],TotalImport*sizeof(double),hipMemcpyHostToDevice,stream1);
	hipStreamSynchronize(stream1);
	//ed = mysecond();
	//fmpi += ed - st;
	//st = mysecond();
	hipStreamSynchronize(stream0);
	//hipDeviceSynchronize();
	//hipMemcpy(&(param->dilatation[numOwnedPoints]),&dilatation[numOwnedPoints],TotalImport*sizeof(double),hipMemcpyHostToDevice);
	//st = mysecond();
	hipLaunchKernelGGL(ForceKernel, dim3(nblocks),dim3(BLOCKSIZE),0,stream0,param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->force, param->neiborlist, numOwnedPoints, bulkModulus,shearModulus,horizon,param->neighborPtr, param->a, param->v, param->u, param->density,param->dt,param->yout,1);
	hipStreamSynchronize(stream0);
	//hipDeviceSynchronize();
	//ed = mysecond();
	//fmpi += ed - st;
	double* temp;
	temp = param->y;
	param->y = param->yout;
	param->yout = temp;
	//ForceTest(x, y, weightvolume, cellvolume, dilatation, bondDamage, force, newNptr, numOwnedPoints, bulkModulus,shearModulus,horizon,newPtr);
	//printf("Force: Kernel %e\n",ed-st);
	//hipMemcpy(y, param->y, (numOwnedPoints)*3*sizeof(double),hipMemcpyDeviceToHost);
	/*st = mysecond();
	hipMemcpy(force, param->force, (numOwnedPoints)*3*sizeof(double),hipMemcpyDeviceToHost);
	ed = mysecond();
	fcpy += ed - st;*/
	//hipStreamDestroy(stream0);
	//hipStreamDestroy(stream1);
	ed = mysecond();
	fcpy += ed - st;
	/*double *dm, *dc, *fm, *fc;
	dm = (double *)malloc(sizeof(double)*mpi_size);
	dc = (double *)malloc(sizeof(double)*mpi_size);
	fc = (double *)malloc(sizeof(double)*mpi_size);
	fm = (double *)malloc(sizeof(double)*mpi_size);
	MPI_Gather(&dmpi, 1, MPI_DOUBLE, dm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dcpy, 1, MPI_DOUBLE, dc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&fmpi, 1, MPI_DOUBLE, fm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&fcpy, 1, MPI_DOUBLE, fc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if(myrank==0){
		double rdm[3],rdc[3], rfm[3], rfc[3];
		rdm[0] = 0;
		rdc[0] = 0;
		rfm[0] = 0;
		rfc[0] = 0;
		rdm[1] = 1.0e10;
		rdc[1] = 1.0e10;
		rfm[1] = 1.0e10;
		rfc[1] = 1.0e10;
		rdm[2] = 0;
		rdc[2] = 0;
		rfm[2] = 0;
		rfc[2] = 0;
		for(i=0;i<mpi_size;i++){
			if(dm[i] > rdm[0])
				rdm[0] = dm[i];
			if(dc[i] > rdc[0])
				rdc[0] = dc[i];
			if(fm[i] > rfm[0])
				rfm[0] = fm[i];
			if(fc[i] > rfc[0])
				rfc[0] = fc[i];
			if(dm[i] < rdm[1])
				rdm[1] = dm[i];
			if(dc[i] < rdc[1])
				rdc[1] = dc[i];
			if(fm[i] < rfm[1])
				rfm[1] = fm[i];
			if(fc[i] < rfc[1])
				rfc[1] = fc[i];
			rdm[2] += dm[i];
			rdc[2] += dc[i];
			rfm[2] += fm[i];
			rfc[2] += fc[i];
		}
		rdm[2] /= mpi_size;	 
		rdc[2] /= mpi_size;	 
		rfm[2] /= mpi_size;	 
		rfc[2] /= mpi_size;	 
		printf("+++++++++++++++++++++++++++++++++\n");
		printf("Dilatation:\n");
		printf("   \tMin\tMAX\tAVG\n");
		printf("MPI\t%e\t%e\t%e\n",rdm[1],rdm[0],rdm[2]);
		printf("CPY\t%e\t%e\t%e\n",rdc[1],rdc[0],rdc[2]);
		printf("Force:\n");
		printf("   \tMin\tMAX\tAVG\n");
		printf("MPI\t%e\t%e\t%e\n",rfm[1],rfm[0],rfm[2]);
		printf("CPY\t%e\t%e\t%e\n",rfc[1],rfc[0],rfc[2]);
		printf("+++++++++++++++++++++++++++++++++\n");
	}*/
	
	//hipMemcpy(dilatation, param->dilatation, (numOwnedPoints+TotalImport)*sizeof(double),hipMemcpyDeviceToHost);
	//printf("After1 myrank %d: %d in dilatation[%d] = %e\n", myrank ,35250, LIDList[nowblock][35250-MinGID], force[0]);
	//printf("Force: Memcpy F %e\n",ed-st);
	//printf("Thanks for using the GPU version!!!! %p,%d\n",param->force,globalnum);
	//const int* nPtr = neighborhoodList;
	//int neighbor = *nPtr;
	//nPtr++;
	/*if(myrank == 1){
	for(int i =0; i<neighbor; i++, nPtr++)
		printf("%e ",dilatation[*nPtr]);
	printf("\n");
	}*/
	
        //Hip
	
}

