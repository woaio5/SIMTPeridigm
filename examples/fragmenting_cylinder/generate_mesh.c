#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double pi;

int main(){
	int k,j,i;
	pi = acos(-1.0);
	double height = 409.600;
	double out_r = 0.025;
	double in_r = 0.020;
	int num_p_r = 5;
	double element_size = (out_r - in_r)/num_p_r;
	int num_p_cir = (int)((2.0*pi*(out_r+in_r)/2.0)/element_size+0.5);
	int num_height =(int)(height/element_size+0.5);
	double arc_length = 2.0*pi/num_p_cir;
	double total_volume = 0.0;
	srand(42);
	FILE* fp = fopen("fragmenting_cylinder4096.txt","w");
	FILE* fp1 = fopen("fragmenting_cylinder_nodeset4096.txt","w");
	int tot = 0;
	//rpcc()
	for(k=0;k<num_height;k++){
		for(j=0;j<num_p_cir;j++){
			for(i=0;i<num_p_r;i++){
				fflush(stdout);
				printf("%d\r",tot);
				double elem_in_r = in_r + i*(out_r - in_r)/num_p_r;
				double elem_out_r = in_r + (i+1)*(out_r-in_r)/num_p_r;
				double alpha = arc_length/2.0;
				double large_dist = 2.0*elem_out_r*sin(alpha)/(3.0*alpha);
				double small_dist = 2.0*elem_in_r*sin(alpha)/(3.0*alpha);
				double large_area = alpha*elem_out_r*elem_out_r;
				double small_area = alpha*elem_in_r*elem_in_r;
				double elem_area = large_area - small_area;
				double elem_dist = (large_dist*large_area-small_dist*small_area)/elem_area;
				double elem_angle = (j + 0.5)*arc_length;
				double elem_height = (k+0.5)*height/num_height;
				double elem_vol = elem_area*height/num_height;
				total_volume += elem_vol;
				double x = elem_dist * cos(elem_angle);
				double y = elem_dist * sin(elem_angle);
				double z = elem_height;
				double magnitude = 0.001 * element_size;
				//x += (2.0*rand() - 1.0) * magnitude;
				//y += (2.0*rand() - 1.0) * magnitude;
				//z += (2.0*rand() - 1.0) * magnitude;
				fprintf(fp,"%.13f %.15f %.15f %d %.11e\n",x,y,z,1,elem_vol);
				fprintf(fp1,"%d\n",tot+1);
				tot++;

					
			}
		}
	}
	fclose(fp);
	
}
