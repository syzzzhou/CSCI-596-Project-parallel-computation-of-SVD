#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"
#include "string.h"
#include "SVD_jacob.h" // Include the header file for SVD_jacob

#define E 0.0001
#define intsize sizeof(int)
#define floatsize sizeof(float)
#define A(x,y) A[x*col+y]                         /* A is matrix row*col */
#define V(x,y) V[x*col+y]                         /* V is matrix col*col */
#define U(x,y) U[x*col+y]
#define B(x)   B[x]
#define a(x,y) a[x*col+y]
#define e(x,y) e[x*col+y]

int col,row;
int m,n,p;
int myid,group_size;
float *A,*V,*B,*U;
float *a,*e;
MPI_Status status;
float starttime,endtime,time1,time2;

void read_fileA()
{
   int i,j;
   FILE *fdA;

   time1=MPI_Wtime();
   fdA=fopen("dataIn.txt","r");
   fscanf(fdA,"%d %d", &row, &col);

   A=(float*)malloc(floatsize*row*col);

   for(i = 0; i < row; i ++)
   {
       for(j = 0; j < col; j ++) fscanf(fdA, "%f", A+i*row+j);
   }
   fclose(fdA);

   printf("Input of file \"dataIn.txt\"\n");
   printf("%d\t %d\n",row, col);
   for(i=0;i<row;i++)
   {
       for(j=0;j<col;j++) printf("%f\t",A(i,j));
       printf("\n");
   }

   V =(float*)malloc(floatsize*col*col);

   for(i=0;i<col;i++)
       for(j=0;j<col;j++)
           if (i==j) V(i,j)=1.0;
   else V(i,j)=0.0;
}

int main(int argc,char **argv)
{
   int loop;
   int i,j,v;
   int p,group_size,myid;
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

    if (myid==0)
        read_fileA(); // Root process reads the input file

    // Broadcast the number of rows and columns to all processes
    MPI_Bcast(&row,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&col,1,MPI_INT,0,MPI_COMM_WORLD);

    // Calculate the number of rows (m) and columns (n) each process will handle
    m=row/p; if (row%p!=0) m++; // For matrix a
    n=col/p; if (col%p!=0) n++; // For matrix e

   if (myid==0)
   {
       B=(float*)malloc(floatsize*col);
       U=(float*)malloc(floatsize*row*col);
   }

   a=(float*)malloc(floatsize*m*col);
   e=(float*)malloc(floatsize*n*col);
   temp=(float*)malloc(floatsize*m);
   temp1=(float*)malloc(floatsize*n);
   ss=(float*)malloc(floatsize*3);
   sum=(float*)malloc(floatsize*3);

   if (myid==0)
   {
       for(i=0;i<m;i++)
           for(j=0;j<col;j++)
               a(i,j)=A(i,j);

       for(i=0;i<n;i++)
           for(j=0;j<col;j++)
               e(i,j)=V(i,j);    // 将矩阵 A 和单位矩阵 I 的部分数据分配给局部矩阵 a 和 e。

       for(i=1;i<p;i++)
       {
           MPI_Send(&(A(m*i,0)),m*col,MPI_FLOAT,i,i,MPI_COMM_WORLD);
           MPI_Send(&(V(n*i,0)),n*col,MPI_FLOAT,i,i,MPI_COMM_WORLD);  //MPI_Send 将矩阵 A 和 V 的部分数据发送给其他进程。
       }
   }
   else
   {
       MPI_Recv(a,m*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD,&status);
       MPI_Recv(e,n*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD,&status);  // 使用 MPI_Recv 接收来自主进程的数据，并存储在局部矩阵 a 和 e 中。
   }

   if (myid==0)                                  /* start  parall computing now */
       time2=MPI_Wtime();

   while (k<=col*(col-1)/2)
   {
       for(i=0;i<col;i++)
           for(j=i+1;j<col;j++)
       {
           ss[0]=0; ss[1]=0; ss[2]=0;
           sum[0]=0; sum[1]=0; sum[2]=0;

           for(v=0;v<m;v++)
               ss[0]=ss[0]+a(v,i)*a(v,j);

           for(v=0;v<m;v++)
               ss[1]=ss[1]+a(v,i)*a(v,i);

           for(v=0;v<m;v++)
               ss[2]=ss[2]+a(v,j)*a(v,j);

           MPI_Allreduce(ss,sum,3,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);  // 使用 MPI_Allreduce 函数将所有进程的 ss 求和，存储在 sum 中。

           if (fabs(sum[0])>E)      // 在旋转矩阵中，只有当 sum[0] 不为 0 时，才需要更新矩阵 a 和 e。
           {
               aa=2*sum[0];
               bb=sum[1]-sum[2];
               rr=sqrt(aa*aa+bb*bb);

               if (bb>=0)
               {
                   c=sqrt((bb+rr)/(2*rr));
                   s=aa/(2*rr*c);
               }
               if (bb<0)
               {
                   s=sqrt((rr-bb)/(2*rr));
                   c=aa/(2*rr*s);
               }

               for(v=0;v<m;v++)
               {
                   temp[v]=c*a(v,i)+s*a(v,j);
                   a(v,j)=(-s)*a(v,i)+c*a(v,j);
               }
               for(v=0;v<m;v++)
                   a(v,i)=temp[v];

               for(v=0;v<n;v++)
               {
                   temp1[v]=c*e(v,i)+s*e(v,j);
                   e(v,j)=(-s)*e(v,i)+c*e(v,j);
               }
               for(v=0;v<n;v++)
                   e(v,i)=temp1[v];
           }
           else
               k++;
       }                                         /* for */
       loop ++;
   }                                             /* while */

   if (myid==0)                 // Collect the data from all processes ,and store them in matrix A and V.
   {
       for(i=0;i<m;i++)
           for(j=0;j<col;j++)
               A(i,j)=a(i,j);

       for(i=0;i<n;i++)
           for(j=0;j<col;j++)
               V(i,j)=e(i,j);
   }

   if (myid!=0)
       MPI_Send(a,m*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD);  // use MPI_Send to send the data to the root process.
   else
   {
       for(j=1;j<p;j++)
       {
           MPI_Recv(a,m*col,MPI_FLOAT,j,j,MPI_COMM_WORLD,&status);

           for(i=0;i<m;i++)
               for(k=0;k<col;k++)
                   A((j*m+i),k)=a(i,k);
       }
   }

   if (myid!=0)
       MPI_Send(e,n*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD);
   else
   {
       for(j=1;j<p;j++)
       {
           MPI_Recv(e,n*col,MPI_FLOAT,j,j,MPI_COMM_WORLD,&status);

           for(i=0;i<n;i++)
               for(k=0;k<col;k++)
                   V((j*n+i),k)=e(i,k);
       }
   }

   if (myid==0)
   {
       for(j=0;j<col;j++)
       {
           su=0.0;
           for(i=0;i<row;i++)
               su=su+A(i,j)*A(i,j);
           B(j)=sqrt(su);
       }

       for(i=1;i<col;i++)
           for(j=0;j<i;j++)
       {
           t=V(i,j);
           V(i,j)=V(j,i);
           V(j,i)=t;
       }

       for(j=0;j<col;j++)
           for(i=0;i<row;i++)
               U(i,j )=A(i,j )/B(j);

       printf(".........U.........\n");
       for(i=0;i<row;i++)
       {
           for(j=0;j<col;j++)
               printf("%f\t",U(i,j));
           printf("\n");
       }

       printf("........E.........\n");
       for(i=0;i<col;i++)
           printf("%f\t",B(i));
       printf("\n");

       printf("........Vt........\n");
       for(i=0;i<col;i++)
       {
           for(j=0;j<col;j++)
               printf("%f\t",V(i,j));
           printf("\n");
       }

       // Call jacob_one_side function from SVD_jacob.h
       double matrix[ROW][COL];
       double I[COL][COL];
       for (int i = 0; i < COL; i++) {
           for (int j = 0; j < COL; j++) {
               I[i][j] = 0;
           }
       }

       for (int i = 0; i < ROW; i++)
           for (int j = 0; j < COL; j++)
               matrix[i][j] = A(i, j);

       jacob_one_side(matrix, I);

       printf("After Jacobian SVD:\n");
       printf("Matrix A:\n");
       for (int i = 0; i < ROW; i++) {
           for (int j = 0; j < COL; j++) {
               printf("%f\t", matrix[i][j]);
           }
           printf("\n");
       }
       printf("Matrix V:\n");
       for (int i = 0; i < COL; i++) {
           for (int j = 0; j < COL; j++) {
               printf("%f\t", I[i][j]);
           }
           printf("\n");
       }
   }

   if (myid==0)
   {
       endtime=MPI_Wtime();

       printf("\n");
       printf("Iteration num = %d\n",loop);
       printf("Whole running time    = %f seconds\n",endtime-starttime);
       printf("Distribute data time  = %f seconds\n",time2-time1);
       printf("Parallel compute time = %f seconds\n",endtime-time2);
   }

   MPI_Finalize();
   free(a);
   free(e);
   free(A);
   free(U);
   free(V);
   free(B);
   free(temp);
   free(temp1);
   return(0);
}