#include "cuda_runtime.h"
#include <iostream>
#include <stdio.h>
#include "gpukernel.h"
#include <math.h>
#define groupsize 4
#define BLOCKSIZE 256
#define FULLMASK 0xffffffff

__global__ void GPU_HelloWorld(int* a){
	int i;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	a[i] = i;
}

__global__ void GPU_ReverseWeightVolume(int numOwnedPoints, double* mp){
	int i,j,k;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<numOwnedPoints){
		mp[i] = 1.0/mp[i];
	}
}

__global__ void GPU_initialize(int numOwnedPoints, double* x, double* v, double* a, double* u, double* y, double dt){
	int i;
	i = blockDim.x * blockIdx.x + threadIdx.x;
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

void GPU_initialize_Interface(int numOwnedPoints, GParam* param){
	int i;
	int nthreads = numOwnedPoints+param->TotalImport;
	int nblocks = (numOwnedPoints+param->TotalImport)/BLOCKSIZE+1;
	GPU_initialize<<<dim3(nblocks), dim3(BLOCKSIZE)>>>(numOwnedPoints, param->x, param->v, param->a, param->u, param->y, param->dt);
	cudaDeviceSynchronize();
}


__global__ void divideNeighbor(int numOwnedPoints, int* neighborhoodlist, int* neighborPtr,int groupnum){
	int i;
	i = blockDim.x * blockIdx.x + threadIdx.x;
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

void divideNeighbor_Interface(int numOwnedPoints, GParam* param){
	int nthreads = numOwnedPoints+param->TotalImport;
	int nblocks = (numOwnedPoints+param->TotalImport)/BLOCKSIZE+1;
	divideNeighbor<<<nblocks,BLOCKSIZE,0,0>>>(numOwnedPoints, param->neiborlist, param->neighborPtr, numOwnedPoints/groupsize+(numOwnedPoints%groupsize!=0));
	cudaDeviceSynchronize();
	GPU_ReverseWeightVolume<<<nblocks,BLOCKSIZE,0,0>>>(numOwnedPoints+param->TotalImport, param->weightvolume);
	cudaDeviceSynchronize();
}


__global__ void DilatationKernel(double *x, double *y, double*m, double* c, double *d, int* bonddamage, int * neighborlist, int numOwnedpoints, double horizon, int* neighborPtr, int outflag){
	int i,j,k;
	i = blockDim.x * blockIdx.x + threadIdx.x;
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
				double xjx = __shfl_sync(FULLMASK,rx[0], stid+(position+k)%groupsize, 32);
				double xjy = __shfl_sync(FULLMASK,rx[1], stid+(position+k)%groupsize, 32);
				double xjz = __shfl_sync(FULLMASK,rx[2], stid+(position+k)%groupsize, 32);
				double cj = __shfl_sync(FULLMASK,rx[3], stid+(position+k)%groupsize, 32);
				double yjx = __shfl_sync(FULLMASK,ry[0], stid+(position+k)%groupsize, 32);
				double yjy = __shfl_sync(FULLMASK,ry[1], stid+(position+k)%groupsize, 32);
				double yjz = __shfl_sync(FULLMASK,ry[2], stid+(position+k)%groupsize, 32);
				double mj = __shfl_sync(FULLMASK,ry[3], stid+(position+k)%groupsize, 32);
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
				dam |= __shfl_sync(FULLMASK,rc[(position+k)%groupsize], stid+(position-k+groupsize)%groupsize, 32);
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

void Dkernel_Interface(GParam* param, int numOwnedPoints, double horizon, cudaStream_t stream0, int flag){
	int nthreads = numOwnedPoints+param->TotalImport;
	int nblocks = (numOwnedPoints+param->TotalImport)/BLOCKSIZE+1;
	DilatationKernel<<<nblocks,BLOCKSIZE,0,stream0>>>(param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->neiborlist, numOwnedPoints, horizon, param->neighborPtr,flag);
}


__global__ void ForceKernel(double *x, double *y, double*m, double* c, double *d, int* bonddamage, double *f, int * neighborlist, int numOwnedpoints,double bulkModulus,double shearModulus, double horizon, int* neighborPtr, double* a, double* v, double* u, double density, double dt, double* yout,int outflag){
	int i,j,k;
	i = blockDim.x * blockIdx.x + threadIdx.x;
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
				double xjx = __shfl_sync(FULLMASK,rx[0], stid+(position+k)%groupsize, 32);
				double xjy = __shfl_sync(FULLMASK,rx[1], stid+(position+k)%groupsize, 32);
				double xjz = __shfl_sync(FULLMASK,rx[2], stid+(position+k)%groupsize, 32);
				double cj = __shfl_sync(FULLMASK,rx[3], stid+(position+k)%groupsize, 32);
				double yjx = __shfl_sync(FULLMASK,ry[0], stid+(position+k)%groupsize, 32);
				double yjy = __shfl_sync(FULLMASK,ry[1], stid+(position+k)%groupsize, 32);
				double yjz = __shfl_sync(FULLMASK,ry[2], stid+(position+k)%groupsize, 32);
				double mj = __shfl_sync(FULLMASK,ry[3], stid+(position+k)%groupsize, 32);
				double dj = __shfl_sync(FULLMASK,rd, stid+(position+k)%groupsize, 32);
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

void Fkernel_Interface(GParam* param, int numOwnedPoints, double horizon, double bulkModulus, double shearModulus, cudaStream_t stream0, int flag){
	int nthreads = numOwnedPoints+param->TotalImport;
	int nblocks = (numOwnedPoints+param->TotalImport)/BLOCKSIZE+1;
	ForceKernel<<<nblocks,BLOCKSIZE,0,stream0>>>(param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->force, param->neiborlist, numOwnedPoints, bulkModulus,shearModulus,horizon,param->neighborPtr, param->a, param->v, param->u, param->density,param->dt,param->yout,flag);
}
