#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "SVD_jacob.h"
#include <omp.h>

int sign(double number)
{
	if (number<0) return -1; else return 1;
}

double vec_mult(double *v1, double *v2,int len)
{
	double val=0;
	//printf("%d%d\n",sizeof(v1),sizeof(v2));
	for (int i=0;i<len;i++)
	{
		val+=(*(v1+i))*(*(v2+i));
	}
	return val;
}

void Orthogonal(double matrix[ROW][COL],int c1, int c2, double V[COL][COL],bool* pass)
{
	double Ci[COL];
	double Cj[COL];
	//printf("%lu   %lu\n",sizeof(Ci),sizeof(Cj));
	for (int i=0;i<ROW;i++)
	{
		Ci[i]=matrix[i][c1];
		Cj[i]=matrix[i][c2];
	}
	double inner_prod=vec_mult(Ci,Cj,ROW);
	if (fabs(inner_prod)<THRESHOLD)
		return;
	*pass=false;
	// for (int i=0;i<ROW;i++)
	// {
	// 	printf("%f  %f\n",Ci[i],Cj[i]);
	// }
	double len1=vec_mult(Ci,Ci,ROW);
	double len2=vec_mult(Cj,Cj,ROW);
	//printf("%f  %f   %f\n",len1,len2,inner_prod);
	if(len1<len2){           
        for(int row=0;row<ROW;++row){
            matrix[row][c1]=Cj[row];
            matrix[row][c2]=Ci[row];
        }
        for(int row=0;row<COL;++row){
            double tmp=V[row][c1];
            V[row][c1]=V[row][c2];
            V[row][c2]=tmp;
        }
    }
	double tao = (len1 - len2) / (2 * inner_prod);
    double tan = sign(tao) / (fabs(tao) + sqrt(1 + pow(tao, 2)));
    double cos = 1 / sqrt(1 + pow(tan, 2));
    double sin = cos * tan;
    for(int row=0;row<ROW;++row){
        double var1=matrix[row][c1]*cos+matrix[row][c2]*sin;
        double var2=matrix[row][c2]*cos-matrix[row][c1]*sin;
        matrix[row][c1]=var1;
        matrix[row][c2]=var2;
    }
    for(int col=0;col<COL;++col){
        double var1=V[col][c1]*cos+V[col][c2]*sin;
        double var2=V[col][c2]*cos-V[col][c1]*sin;
        V[col][c1]=var1;
        V[col][c2]=var2;
	}
}

void jacob_one_side(double matrix[ROW][COL], double V[COL][COL])
{
	int iterat=ITERATION;
	while (iterat-->0)
	{
		bool pass = true;
        #pragma omp parallel
        #pragma omp for
        for (int i = 0; i < COL; ++i) {
            for (int j = i+1; j<COL; ++j) {
                Orthogonal(matrix, i, j, V, &pass);
            }
        }
        printf("%d\n",ITERATION-iterat);
        if (pass) 
        {
        	//printf("%s\n","cnm");
            break;
        }
	}
}

int main(int argc, char **argv)
{
    if (ROW > COL)
    {
    	return 0;
    }
    double A[ROW][COL];
    double vec[]= {6,5,1,9,8,4,8,5,2,4,6,9,1,2,3,2,1,4};
    double V[COL][COL];
    //printf("V=");
    //print_matrix((double *)V,COL,COL);
    double S[ROW][COL];
    omp_set_num_threads(4);
    double U[ROW][ROW];
    
    //for (int i=0;i<COL;i++)
    //{
       // V[i][i]=1.0;
    //}
    for (int i=0;i<ROW;i++)
    {
    	for (int j=0;j<COL;j++)
    	{
    		A[i][j]=vec[i*COL+j];
    	}
    }
    printf("A=");
    print_matrix((double *)A,ROW,COL);
    //printf("V=");
    //print_matrix((double *)V,COL,COL);
    for (int i=0;i<COL;i++)
    {
        for (int j=0;j<COL;j++){
	
    	if (i==j) V[i][i]=1; else V[i][j]=0;
        }
    }
    // printf("V=");
    // print_matrix((double *)V,COL,COL);
    jacob_one_side(A,V);
    double E[COL];
    for (int i=0;i<COL;i++)
    {
        E[i]=0;
    }
    int nonzero=0;
    for (int i = 0; i < COL; ++i) {
    	double norm=0;
        for (int j=0;j<ROW;j++)
        {
        	norm+=A[j][i]*A[j][i];
        }
        if (norm>THRESHOLD)
        	nonzero+=1;
        E[i]=sqrt(norm);          
    }
    for (int i = 0; i < ROW; ++i) {
	for (int j=0;j<COL;++j){
    	if (i==j) S[i][i]=E[i]; else S[i][j]=0;
        }
    }
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < ROW; ++j){
        U[i][j]=0;
       }
    }
    for (int i=0;i<ROW;i++)
    {
        for (int j=0;j<nonzero;j++)
        {
        	U[i][j]=A[i][j]/E[j];
        }      
    } 
    //printf("%f.  %f\n",E[0],S[0][0]);
    printf("A=");
    print_matrix((double *)A,ROW,COL);
    printf("S=");
    print_matrix((double *)S,ROW,COL);
    printf("V=");
    print_matrix((double *)V,COL,COL);
    printf("U=");
    print_matrix((double *)U,ROW,ROW);
}



