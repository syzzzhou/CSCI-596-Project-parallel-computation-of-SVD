// To generate the dataIn.txt file, set the matrix dimension, and fill the matrix with random float numbers between 0 and 1.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int row = 500;
    int col = 500;
    FILE *fp = fopen("dataIn.txt", "w");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    // write row and col
    fprintf(fp, "%d %d\n", row, col);

    // random number generator
    srand((unsigned)time(NULL));

    // put number into the matrix
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            // generate random float number between 0 and 1
            float val = (float)rand() / (float)RAND_MAX;
            fprintf(fp, "%f", val);
            if (j < col - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;
}
