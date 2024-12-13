#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"
#include "string.h"
#include "SVD_jacob.h" // Include the header file for SVD_jacob

#define E 0.0001
#define intsize sizeof(int)
#define floatsize sizeof(float)

#define A_IDX(x,y) A[(x)*col+(y)]
#define V_IDX(x,y) V[(x)*col+(y)]
#define U_IDX(x,y) U[(x)*col+(y)]
#define B_IDX(x)   B[(x)]
#define a_IDX(x,y) a[(x)*col+(y)]
#define e_IDX(x,y) e[(x)*col+(y)]

int col,row;
int m,n,p;
int myid,group_size;
float *A,*V,*B,*U;
float *a,*e;
MPI_Status status;
float starttime,endtime,time1,time2;

// no need to read file
// void read_fileA() { ... }

int main(int argc,char **argv)
{
   int loop;
   int i,j,v;
   int k;
   float *sum, *ss;
   float aa,bb,rr,c,s,t;
   float su;
   float *temp,*temp1;

   MPI_Init(&argc,&argv);
   MPI_Comm_size(MPI_COMM_WORLD,&group_size);
   MPI_Comm_rank(MPI_COMM_WORLD,&myid);

   if (myid==0) starttime=MPI_Wtime(); // Start the timer on the root process
   p=group_size; // Set the number of processes
   loop=0; // Initialize loop counter
   k=0; // Initialize k counter

   // set the matrix dimension
   int baseN = 500;
   double scale_factor = pow((double)p/1.0, 1.0/3.0);
   row = col = (int)(baseN * scale_factor);

   if (myid==0) {
       A = (float *)malloc(floatsize * row * col);
       srand(0);
       for (i = 0; i < row; i++) {
           for (j = 0; j < col; j++) {
               // inialize A with random float number between 0 and 1
               A_IDX(i,j) = (float)rand() / (float)RAND_MAX;
           }
       }

       V = (float *)malloc(floatsize * col * col);
       for (i = 0; i < col; i++) {
           for (j = 0; j < col; j++) {
               if (i == j)
                   V_IDX(i,j) = 1.0;
               else
                   V_IDX(i,j) = 0.0;
           }
       }
   }
   // =========== Weak Scaling ===========

   time1 = MPI_Wtime(); // start the timer for data distribution

   // Broadcast the number of rows and columns to all processes
   MPI_Bcast(&row,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&col,1,MPI_INT,0,MPI_COMM_WORLD);

   int rows_per_proc = row / p;
   int remainder_rows = row % p;
   int cols_per_proc = col / p;
   int remainder_cols = col % p;

   int local_m = rows_per_proc + (myid < remainder_rows ? 1 : 0);
   int local_n = cols_per_proc + (myid < remainder_cols ? 1 : 0);

   int *all_m = NULL;
   int *all_n = NULL;
   int *row_displs = NULL;
   int *col_displs = NULL;

   if (myid == 0) {
       all_m = (int*)malloc(p*sizeof(int));
       all_n = (int*)malloc(p*sizeof(int));
       for (int r=0; r<p; r++) {
           all_m[r] = rows_per_proc + (r < remainder_rows ? 1 : 0);
           all_n[r] = cols_per_proc + (r < remainder_cols ? 1 : 0);
       }
       row_displs = (int*)malloc(p*sizeof(int));
       col_displs = (int*)malloc(p*sizeof(int));

       row_displs[0]=0;
       for (int r=1; r<p; r++) {
           row_displs[r]=row_displs[r-1]+all_m[r-1];
       }

       col_displs[0]=0;
       for (int r=1; r<p; r++) {
           col_displs[r]=col_displs[r-1]+all_n[r-1];
       }
   }

   if (myid==0)
   {
       B=(float*)malloc(floatsize*col);
       U=(float*)malloc(floatsize*row*col);
   }

   a=(float*)malloc(floatsize*local_m*col);
   e=(float*)malloc(floatsize*local_n*col);
   temp=(float*)malloc(floatsize*local_m);
   temp1=(float*)malloc(floatsize*local_n);
   ss=(float*)malloc(floatsize*3);
   sum=(float*)malloc(floatsize*3);

   // data distribution
   if (myid==0)
   {
       // copy the part belonging to process 0
       for(i=0;i<local_m;i++)
           for(j=0;j<col;j++)
               a_IDX(i,j)=A_IDX(row_displs[0]+i,j);

       for(i=0;i<local_n;i++)
           for(j=0;j<col;j++)
               e_IDX(i,j)=V_IDX(col_displs[0]+i,j);

       // send the remaining parts to other processes
       for (int r=1; r<p; r++) {
           MPI_Send(A + (row_displs[r]*col), all_m[r]*col, MPI_FLOAT, r, r, MPI_COMM_WORLD);
           MPI_Send(V + (col_displs[r]*col), all_n[r]*col, MPI_FLOAT, r, r, MPI_COMM_WORLD);
       }
   }
   else
   {
       MPI_Recv(a,local_m*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD,&status);
       MPI_Recv(e,local_n*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD,&status);
   }

   if (myid==0)
       time2=MPI_Wtime();

   while (k<=col*(col-1)/2)
   {
       for(i=0;i<col;i++)
           for(j=i+1;j<col;j++)
       {
           ss[0]=0; ss[1]=0; ss[2]=0;
           sum[0]=0; sum[1]=0; sum[2]=0;

           for(v=0;v<local_m;v++)
               ss[0]=ss[0]+a_IDX(v,i)*a_IDX(v,j);

           for(v=0;v<local_m;v++)
               ss[1]=ss[1]+a_IDX(v,i)*a_IDX(v,i);

           for(v=0;v<local_m;v++)
               ss[2]=ss[2]+a_IDX(v,j)*a_IDX(v,j);

           MPI_Allreduce(ss,sum,3,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

           if (fabs(sum[0])>E)
           {
               aa=2*sum[0];
               bb=sum[1]-sum[2];
               rr=sqrt(aa*aa+bb*bb);

               if (bb>=0)
               {
                   c=sqrt((bb+rr)/(2*rr));
                   s=aa/(2*rr*c);
               }
               else
               {
                   s=sqrt((rr-bb)/(2*rr));
                   c=aa/(2*rr*s);
               }

               for(v=0;v<local_m;v++)
               {
                   temp[v]=c*a_IDX(v,i)+s*a_IDX(v,j);
                   a_IDX(v,j)=(-s)*a_IDX(v,i)+c*a_IDX(v,j);
               }
               for(v=0;v<local_m;v++)
                   a_IDX(v,i)=temp[v];

               for(v=0;v<local_n;v++)
               {
                   temp1[v]=c*e_IDX(v,i)+s*e_IDX(v,j);
                   e_IDX(v,j)=(-s)*e_IDX(v,i)+c*e_IDX(v,j);
               }
               for(v=0;v<local_n;v++)
                   e_IDX(v,i)=temp1[v];
           }
           else
               k++;
       }
       loop ++;
   }

   // Gather the data
   if (myid==0)
   {
       for(i=0;i<local_m;i++)
           for(j=0;j<col;j++)
               A_IDX(row_displs[0]+i,j)=a_IDX(i,j);

       for(i=0;i<local_n;i++)
           for(j=0;j<col;j++)
               V_IDX(col_displs[0]+i,j)=e_IDX(i,j);
   }

   if (myid!=0)
   {
       MPI_Send(a,local_m*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD);
   }
   else
   {
       for(int r=1;r<p;r++)
       {
           MPI_Recv(a,all_m[r]*col,MPI_FLOAT,r,r,MPI_COMM_WORLD,&status);
           for(i=0;i<all_m[r];i++)
               for(k=0;k<col;k++)
                   A_IDX(row_displs[r]+i,k)=a_IDX(i,k);
       }
   }

   if (myid!=0)
   {
       MPI_Send(e,local_n*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD);
   }
   else
   {
       for(int r=1;r<p;r++)
       {
           MPI_Recv(e,all_n[r]*col,MPI_FLOAT,r,r,MPI_COMM_WORLD,&status);

           for(i=0;i<all_n[r];i++)
               for(k=0;k<col;k++)
                   V_IDX(col_displs[r]+i,k)=e_IDX(i,k);
       }
   }

   if (myid==0)
   {
       for(j=0;j<col;j++)
       {
           su=0.0;
           for(i=0;i<row;i++)
               su=su+A_IDX(i,j)*A_IDX(i,j);
           B_IDX(j)=sqrt(su);
       }

       for(i=1;i<col;i++)
           for(j=0;j<i;j++)
       {
           t=V_IDX(i,j);
           V_IDX(i,j)=V_IDX(j,i);
           V_IDX(j,i)=t;
       }

       for(j=0;j<col;j++)
           for(i=0;i<row;i++)
               U_IDX(i,j)=A_IDX(i,j)/B_IDX(j);

       int ROW = row;
       int COL = col;
       double (*matrix)[COL] = malloc(sizeof(double)*ROW*COL);
       double (*I)[COL] = malloc(sizeof(double)*COL*COL);

       for (int x = 0; x < COL; x++)
           for (int y = 0; y < COL; y++)
               I[x][y] = 0.0;

       for (int x = 0; x < ROW; x++)
           for (int y = 0; y < COL; y++)
               matrix[x][y] = A_IDX(x, y);

       jacob_one_side(matrix, I);

       free(matrix);
       free(I);

       endtime=MPI_Wtime();
       printf("\n========== Isogranular Scaling Test Results ==========\n");
       printf("Number of processes  = %d\n",group_size);
       printf("Matrix dimension     = %d x %d (Isogranular scaled)\n",row,col);
       printf("Iteration num        = %d\n",loop);
       printf("Whole running time   = %f seconds\n",endtime-starttime);
       printf("Distribute data time = %f seconds\n",time2-time1);
       printf("Parallel compute time= %f seconds\n",endtime-time2);
       printf("=====================================================\n");
   }

   free(a);
   free(e);
   free(temp);
   free(temp1);
   free(ss);
   free(sum);

   if (myid == 0) {
       free(A);
       free(U);
       free(V);
       free(B);
       free(all_m);
       free(all_n);
       free(row_displs);
       free(col_displs);
   }

   MPI_Finalize();
   return(0);
}
