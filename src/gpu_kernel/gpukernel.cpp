#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "gpukernel.h"
#include "mpi.h"
#include "kernelfunc.h"
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
cudaStream_t stream0, stream1;

double mysecond(){
        struct timeval tp;
        struct timezone tpz;
        int i = gettimeofday(&tp, &tpz);
        return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


void InitVUY(GParam* param, int numOwnedPoints){
	int i;
	int nthreads = numOwnedPoints+param->TotalImport;
	int nblocks = (numOwnedPoints+param->TotalImport)/BLOCKSIZE+1;
	GPU_initialize_Interface(numOwnedPoints, param);
	//GPU_initialize,dim3(nblocks), dim3(BLOCKSIZE),numOwnedPoints, param->x, param->v, param->a, param->u, param->y, param->dt,0,0);
	cudaDeviceSynchronize();
}


void InitDeviceMemory(GParam* param,int numOwnedPoints, int neighborsize, double*x , double *y,double* weightvolume, double* cellvolume, int* bonddamage, int* neighborhoodlist, int* neighborPtr,int globalnum){
	int i;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaMalloc((void**)&(param->x),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	cudaMalloc((void**)&(param->y),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	cudaMalloc((void**)&(param->v),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	cudaMalloc((void**)&(param->a),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	cudaMalloc((void**)&(param->u),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	cudaMalloc((void**)&(param->force),(numOwnedPoints+param->TotalImport)*3*sizeof(double));
	cudaMalloc((void**)&(param->weightvolume),(numOwnedPoints+param->TotalImport)*sizeof(double));
	cudaMalloc((void**)&(param->yout), (numOwnedPoints+param->TotalImport)*3*sizeof(double));
	//cudaMalloc((void**)&(param->density),globalnum/3*sizeof(double));
	cudaMalloc((void**)&(param->cellvolume),(numOwnedPoints+param->TotalImport)*sizeof(double));
	cudaMalloc((void**)&(param->dilatation),(numOwnedPoints+param->TotalImport)*sizeof(double));
	cudaMalloc((void**)&(param->neighborPtr),numOwnedPoints*sizeof(int));
	cudaMalloc((void**)&(param->mybondDamage),neighborsize*sizeof(int));
	cudaMalloc((void**)&(param->neiborlist),neighborsize*sizeof(int));
	cudaMemcpy(param->x, x, (numOwnedPoints+param->TotalImport)*3*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(param->y, y, (numOwnedPoints+param->TotalImport)*3*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(param->cellvolume, cellvolume, (numOwnedPoints+param->TotalImport)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(param->weightvolume, weightvolume, (numOwnedPoints+param->TotalImport)*sizeof(double),cudaMemcpyHostToDevice);
	//cudaMemcpy(param->density, density, globalnum/3*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(param->mybondDamage, bonddamage, (neighborsize)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(param->neiborlist, neighborhoodlist, neighborsize*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(param->neighborPtr, neighborPtr, (numOwnedPoints/4+(numOwnedPoints%groupsize!=0))*sizeof(int),cudaMemcpyHostToDevice);
	//newNptr = (int *)malloc(sizeof(int)*neighborsize);
	//newPtr = (int *)malloc(sizeof(int)*numOwnedPoints/4+(numOwnedPoints%groupsize!=0));
	mybondDamage = (int *)malloc(sizeof(int)*(neighborsize));
	int deviceID = myrank % 4;
    cudaGetDevice(&deviceID);
	int nthreads = numOwnedPoints+param->TotalImport;
	int nblocks = (numOwnedPoints+param->TotalImport)/BLOCKSIZE+1;
	//int globalnum = param->numOwnedpoints;
	printf("Initial Memory on Device %e %e %d\n",param->dt, x[0],param->TotalImport);
	//hipLaunchKernelGGL(GPU_initialize, dim3(nblocks),dim3(BLOCKSIZE),0,0,numOwnedPoints, param->x, param->v, param->a, param->u, param->y, param->dt);
	//cudaDeviceSynchronize();
	divideNeighbor_Interface(numOwnedPoints, param);
    //divideNeighbor<<<nblocks,BLOCKSIZE,0,0>>>(numOwnedPoints, param->neiborlist, param->neighborPtr, numOwnedPoints/groupsize+(numOwnedPoints%groupsize!=0));
	//cudaDeviceSynchronize();

	//GPU_ReverseWeightVolume<<<nblocks,BLOCKSIZE,0,0>>>(numOwnedPoints+param->TotalImport, param->weightvolume);
	//cudaDeviceSynchronize();
	//cudaMemcpy(newNptr, param->neiborlist,neighborsize*sizeof(int),cudaMemcpyDeviceToHost);
	//cudaMemcpy(newPtr, param->neighborPtr, (numOwnedPoints/4+(numOwnedPoints%groupsize!=0))*sizeof(int),cudaMemcpyDeviceToHost);
	int st = neighborPtr[0];
	//for(int i=0; i< neighborhoodlist[st];i++)
		printf("inner:%d outter:%d\n",(neighborhoodlist[0]&0x0000ffff),(neighborhoodlist[0]&0xffff0000)>>16);
	//printf("\n");
	
}

void CPU_HelloWorld(int *a){
	int i;
	int *deva;
	//cudaMalloc((void **)&deva, 1024*sizeof(int));
	//printf("~~~~CPU HELLOWOLRD!!!\n");
	for(i=0;i<1024;i++)
		a[i] = 0;
	/*cudaMemcpy(deva, a, 1024*sizeof(int), cudaMemcpyHostToDevice);
	hipLaunchKernelGGL(GPU_HelloWorld, dim3(8),dim3(128),0,0,deva);
	cudaMemcpy(a, deva, 1024*sizeof(int), cudaMemcpyDeviceToHost);
	if(a[128]==128)
		printf("~~~GPU is started successfully!!!!!\n");*/
}





void ForceTest(double *x, double *y, double*m, double* c, double *d, double* bonddamage, double *f, const int * neighborlist, int numOwnedpoints,double bulkModulus,double shearModulus, double horizon, int* neighborPtr){
	int i = 0;
	//i = blockDim.x * blockId.x + threadId.x;
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

	//st = mysecond();
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
	Dkernel_Interface(param, numOwnedPoints, horizon, stream0,0);
	//DilatationKernel<<<nblocks,BLOCKSIZE,0,stream0>>>(param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->neiborlist, numOwnedPoints, horizon, param->neighborPtr,0);
	cudaMemcpyAsync(y, param->y, numOwnedPoints*3*sizeof(double), cudaMemcpyDeviceToHost, stream1);
	cudaStreamSynchronize(stream1);
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
	cudaMemcpy(param->y,hostd,globalnum*sizeof(double),cudaMemcpyHostToDevice);
	ed = mysecond();
	printf("DIlatation: Memcpy Pined Y %e\n",ed-st);*/
	//int **NumImport = param->NumImport;
	int TotalImport = 0;
	for(i=0;i<mpi_size;i++){
		TotalImport += NumImport[nowblock][i];
	}
	//cudaMemcpy(param->y,y,globalnum*sizeof(double),cudaMemcpyHostToDevice);
	//printf("DIlatation: Memcpy Y %e, %d\n",ed-st, numOwnedPoints);
	//cudaMemcpy(param->bonddamage, param->mybondDamage, (param->neighborsize-numOwnedPoints/groupsize)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpyAsync(&(param->y[numOwnedPoints*3]),&y[numOwnedPoints*3],(TotalImport)*3*sizeof(double),cudaMemcpyHostToDevice,stream1);
	cudaStreamSynchronize(stream1);
	//dmpi += ed - st;
	//cudaDeviceSynchronize();
	//st = mysecond();
	cudaStreamSynchronize(stream0);
	//ed = mysecond();
	//cudaMemcpy(&(param->y[numOwnedPoints*3]),&y[numOwnedPoints*3],(TotalImport)*3*sizeof(double),cudaMemcpyHostToDevice);
	//st = mysecond();
	Dkernel_Interface(param, numOwnedPoints, horizon, stream0,1);
//	DilatationKernel<<<nblocks,BLOCKSIZE,0,stream0>>>(param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->neiborlist, numOwnedPoints, horizon, param->neighborPtr,1);
	cudaStreamSynchronize(stream0);
	//acudaDeviceSynchronize();
	//ed = mysecond();
	//dmpi += ed - st;
	//printf("DIlatation: Kernel %e\n",ed-st);
	/*cudaMemcpy( mybondDamage ,param->mybondDamage, (param->neighborsize)*sizeof(int),cudaMemcpyDeviceToHost);
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
	//cudaMemcpy(&dilatation[numOwnedPoints], &((param->dilatation)[numOwnedPoints]), TotalImport*sizeof(double),cudaMemcpyDeviceToHost);
	//cudaMemcpy(dilatation, param->dilatation, numOwnedPoints*sizeof(double),cudaMemcpyDeviceToHost);
	//ed = mysecond();
	//dcpy += ed -st;
	//cudaStreamDestroy(stream0);
	//cudaStreamDestroy(stream1);
	//ed = mysecond();
	//dcpy += ed - st;
	//char result[1000];
	//sprintf(result,"+++++++++++++++++++++++++\nDilatation Kernel BreakDown of %d\nMPI:%e\nMemcpy:%e\nKernel:%e\n+++++++++++++++++++++++++\n",myrank,mpi,cpy,ken);
	//printf("%s",result);
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
	//st = mysecond();
	//cudaStream_t stream0, stream1;
	//cudaStreamCreate(&stream0);
	//cudaStreamCreate(&stream1);
	int TotalImport = 0;
	for(i=0;i<mpi_size;i++){
		TotalImport += NumImport[nowblock][i];	
	}
	//printf("Before myrank %d: %d in dilatation[%d] = %e\n", myrank ,35250, LIDList[nowblock][35250-MinGID], dilatation[35250]);
	//st = mysecond();
	Fkernel_Interface(param, numOwnedPoints, horizon, bulkModulus, shearModulus, stream0,0);

	//ForceKernel<<<nblocks,BLOCKSIZE,0,stream0>>>(param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->force, param->neiborlist, numOwnedPoints, bulkModulus,shearModulus,horizon,param->neighborPtr, param->a, param->v, param->u, param->density,param->dt,param->yout,0);
	cudaMemcpyAsync(dilatation, param->dilatation, numOwnedPoints*sizeof(double), cudaMemcpyDeviceToHost, stream1);
	cudaStreamSynchronize(stream1);
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
	//cudaMemcpy(param->y,y,globalnum*sizeof(double),cudaMemcpyHostToDevice);
	//cudaMemcpy(param->weightvolume,weightvolume,globalnum/3*sizeof(double),cudaMemcpyHostToDevice);
	//cudaMemcpy(param->dilatation,dilatation,globalnum/3*sizeof(double),cudaMemcpyHostToDevice);
	//ed = mysecond();
	//printf("Force: Memcpy Y and D %e\n",ed-st);
	int *neighborPtr = (int *)malloc(sizeof(int)*numOwnedPoints);
	/*cudaMemcpy(force, param->force, (numOwnedPoints)*3*sizeof(double),cudaMemcpyDeviceToHost);
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
	//cudaMemcpy(neighborPtr, param->neighborPtr, (numOwnedPoints)*sizeof(int ),cudaMemcpyDeviceToHost);
	//printf("Thanks for using the GPU version!!!! %p,%d\n",param->force,globalnum);
	//printf("After myrank %d: %d in dilatation[%d] = %e\n", myrank ,35250, LIDList[nowblock][35250-MinGID], weightvolume[35250]);
	//param->numOwnedpoints = numOwnedPoints;
	//st = mysecond();
	cudaMemcpyAsync(&(param->dilatation[numOwnedPoints]),&dilatation[numOwnedPoints],TotalImport*sizeof(double),cudaMemcpyHostToDevice,stream1);
	cudaStreamSynchronize(stream1);
	//ed = mysecond();
	//fmpi += ed - st;
	//st = mysecond();
	cudaStreamSynchronize(stream0);
	//ed = mysecond();
	//cudaDeviceSynchronize();
	//cudaMemcpy(&(param->dilatation[numOwnedPoints]),&dilatation[numOwnedPoints],TotalImport*sizeof(double),cudaMemcpyHostToDevice);
	//st = mysecond();
   	Fkernel_Interface(param, numOwnedPoints, horizon, bulkModulus, shearModulus, stream0,1);

    //ForceKernel<<<nblocks,BLOCKSIZE,0,stream0>>>(param->x, param->y, param->weightvolume, param->cellvolume, param->dilatation, param->mybondDamage, param->force, param->neiborlist, numOwnedPoints, bulkModulus,shearModulus,horizon,param->neighborPtr, param->a, param->v, param->u, param->density,param->dt,param->yout,1);
	cudaStreamSynchronize(stream0);
	//cudaDeviceSynchronize();
	//ed = mysecond();
	//fmpi += ed - st;
	double* temp;
	temp = param->y;
	param->y = param->yout;
	param->yout = temp;
	//ForceTest(x, y, weightvolume, cellvolume, dilatation, bondDamage, force, newNptr, numOwnedPoints, bulkModulus,shearModulus,horizon,newPtr);
	//printf("Force: Kernel %e\n",ed-st);
	//cudaMemcpy(y, param->y, (numOwnedPoints)*3*sizeof(double),cudaMemcpyDeviceToHost);
	/*st = mysecond();
	cudaMemcpy(force, param->force, (numOwnedPoints)*3*sizeof(double),cudaMemcpyDeviceToHost);
	ed = mysecond();
	fcpy += ed - st;*/
	//cudaStreamDestroy(stream0);
	//cudaStreamDestroy(stream1);
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
	
	//cudaMemcpy(dilatation, param->dilatation, (numOwnedPoints+TotalImport)*sizeof(double),cudaMemcpyDeviceToHost);
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

