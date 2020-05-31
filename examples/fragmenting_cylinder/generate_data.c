#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define CSIZE 4

int mid[70500][1000];
int tail[2821][1000];
int head[70500][1000];

long long sizehead, sizemid, sizetail;
long long total;
void process(char *line,int n,int* mid, long long* size){
	int i = 0,j;
	int numcomma = 0;
	int prev = 0;
	char number[20];
	while(line[i] != '\n'){
		if(line[i] == ','){
			for(j=0;j<20;j++)
				number[j] = '\0';
			if(numcomma == 0){
				memcpy(number,&line[prev],i-prev);
				mid[numcomma] = atoi(number);
				*size += mid[numcomma];
				prev = i+1;
			}else{
				memcpy(number,&line[prev],i-prev);
				mid[numcomma] = atoi(number);
				prev= i+1;
			}
			total++;
			numcomma++;
		}
		i++;	
	}
}


int main(){
	FILE *fp1, *fp2,*fpm1,*fpm2;
	int numPoints,numPoints2;
	int size1, size2;
	int p1,p2,i,j,k;
	sizehead = 0;
	sizemid = 0;
	sizetail = 0;
	total = 0;
	int n=8192;
	char line1[2000],line2[2000];
	fp1 = fopen("neighborlist_1.txt","r");
	fp2 = fopen("neighborlist_2.txt","r");
	//fpm1 = fopen("newmesh_ori_36160.txt","r");
	//fpm2 = fopen("newmesh_ori_ext2.txt","r");
	
	fscanf(fp1,"%d\n",&size1);
	fscanf(fp2,"%d\n",&size2);
	fscanf(fp1,"%d\n",&numPoints);
	fscanf(fp2,"%d\n",&numPoints2);
	//printf("head:[0,%d),mid1:[%d,%d),mid2:[%d,%d),end[%d,%d)\n",p1,p1,p2,p2,numPoints/CSIZE+p1,numPoints/CSIZE+p1,numPoints2/CSIZE);
	for(i=0;i<numPoints;i++){
		fgets(line1,2000,fp1);
		fgets(line2,2000,fp2);
		if(strcmp(line1,line2)!=0)
			break;
	}
	fclose(fp1);
	fclose(fp2);
	//fclose(fpm1);
	//fclose(fpm2);
	//fp1 = fopen("neiborlist_ori4_36160.txt","r");
	fp2 = fopen("neighborlist_2.txt","r");
	int nouse;
	fscanf(fp2,"%d\n",&nouse);
	fscanf(fp2,"%d\n",&nouse);
	p1 = i;
	p2 = numPoints+(numPoints-p1);
	FILE* fout = fopen("neighborlist_8192.txt","w");
	//fprintf(fout,"%d\n",nouse*n/2);
//head	
	for(i=0;i<p1;i++){
		//fgets(line1,2000,fp1);
		fgets(line2,2000,fp2);
		process(line2,i,head[i], &sizehead);
		//fprintf(fout,"%s",line2);
	}
	//printf("total after head:%d\n",total);
//mid
	//for(i=0;i<n-1;i++){
		for(k=0;k<70500;k++){
			fgets(line2,2000,fp2);
			process(line2,k,mid[k], &sizemid);
			//fprintf(fout,"%d,",temp[0]);
			//for(j=0;j<temp[0];j++)
			//	fprintf(fout,"%d,",temp[j]);
			//fprintf(fout,"\n");
		}
	//printf("total after mid:%d\n",total);
	for(i=70500+p1;i<70500*2;i++){
			fgets(line2,2000,fp2);
			j = i - (70500+p1);
			process(line2,j,tail[j], &sizetail);
		
	}
	fclose(fp2);
		
	//}
//tail
	/*for(i=p1+36160/CSIZE;i<36160*n/CSIZE;i++){
		
	}*/
	printf("!!!!!%lld %lld %lld %lld %lld\n",sizehead, sizemid, sizetail, sizehead+sizemid*(n-1)+sizetail+70500*n, total);
	total = sizehead+sizemid*(n-1)+sizetail+70500*n;
	fprintf(fout,"%lld\n",total);
	fprintf(fout,"%d\n",70500*n);
	for(i=0;i<p1;i++){
		fprintf(fout, "%d,", head[i][0]);
		for(j=1;j<=head[i][0];j++){
			fprintf(fout,"%d,",head[i][j]);
		}
		fprintf(fout,"\n");
	}
	for(i=0;i<n-1;i++){
		fflush(stdout);
		printf("%d\r",i);
		for(j = 0;j < 70500;j++){
			fprintf(fout,"%d,",mid[j][0]);
			for(k=1;k<=mid[j][0];k++){
				fprintf(fout,"%d,",mid[j][k]+70500*i);
			}
			fprintf(fout,"\n");
		}
		
	}
	for(i=0;i<70500*2-(70500+p1);i++){
		fprintf(fout,"%d,",tail[i][0]);
		for(j=1;j<=tail[i][0];j++){
			fprintf(fout,"%d,",tail[i][j]+70500*(n-2));
		}
		fprintf(fout,"\n");
	}
	return 0;	
	
	


	
}

