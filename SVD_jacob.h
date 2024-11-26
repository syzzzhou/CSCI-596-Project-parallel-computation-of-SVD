double THRESHOLD=1E-8;
int ITERATION=30;
int ROW=2;
int COL=3;

void print_matrix(double *A,int r,int c)
{
	for (int i=0;i<r;i++)
    {
    	for (int j=0;j<c;j++)
    	{
    		printf("%f  ",*((A+i*c) + j));
    	}
    	printf("\n");
    }
}


