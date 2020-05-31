#include <sys/time.h>


typedef struct GPU_Parameter_ELASTIC{
	double* x;
	double* y;
	double* weightvolume;
	double* cellvolume;
	double* dilatation;
	double* bonddamage;
	double* force;
	double density;
	double dt;
	double* v;
	double* a;
	double* u;
	double* partialStress;
	int* neiborlist;
	int* neighborPtr;
	int* myneighborlist;
	int* mybondDamage;
	int* myPtr;
	int neighborsize;
	int numOwnedpoints;
	double bulkModulus;
	double shearModulus;
	double horizon;
	double alpha;
	int TotalImport;
	double* deltaTemprature;
	double ***sendbuf;
	double ***recvbuf;
	int **NumProcRequest;
	int **NumImport;
	int ***ProcRequestList;
	int ***ProcExportList;
	int **LIDList;
	int MaxGID;
	int MinGID;
	double* yout;
}GParam;

typedef struct Communication_Param{
	int **ProcIDList;
}CParam;

extern GParam *gparamlist;
extern CParam *cparamlist;
extern int myrank;


extern int Initflag;
extern int nowblock;

//extern double *force1;
//extern double *dilatation1;

//void InitDeviceMemory(GParam* param,int numOwnedPoints, int neighborsize, double* x ,double* weightvolume, double* cellvolume, double* bonddamage, int* neighborhoodlist,int* neighborPtr,int globalnum);
//void InitDeviceMemory(GParam* param,int numOwnedPoints, int neighborsize, double* x ,double* weightvolume, double* cellvolume, int* bonddamage, int* neighborhoodlist,int* neighborPtr,int globalnum);
void InitDeviceMemory(GParam* param,int numOwnedPoints, int neighborsize, double* x ,double* y,double* weightvolume, double* cellvolume, int* bonddamage, int* neighborhoodlist,int* neighborPtr,int globalnum);
//void InitD$eviceMemory(GParam* param,int numOwnedPoints, int neighborsize, double* x ,double* y,double* weightvolume, double* cellvolume, int* bonddamage, int* neighborhoodlist,int* neighborPtr,int globalnum, double* density);
void InitVUY(GParam* param, int numOwnedPoints);
