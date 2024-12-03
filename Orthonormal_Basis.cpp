#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

/* ARGUMENT STRUCTS */

struct detHelperArgs {
    double **matrix;
    int size, v;
    double det;
};

struct normHelperArgs {
    double *v;
    int size;
};


/* FUNCTION PROTOTYPES */

// Utility functions
void printMatrix(double **matrix, int row, int col, bool nonZero = true);   // Prints a matrix to the screen
double dotProduct(double *a, double *b, int size);                          // Calculates the dot product of two vectors
double **transpose(double **matrix, int row, int col);                      // Transposes a matrix (swaps rows and columns)
void swapVectors(double *a, double *b, int size);                           // Swaps all of the values of two vectors

// Step 1: Linear Independence
bool linearIndependence(double **matrix, int row, int col); // Checks if the columns of a matrix are linearly independent

// Row reduction to Row Echelon Form
double **rref(double **matrix, int row, int col);   // Calculates the reduced row-echelon form of a matrix

// Determinant using cofactor expansion (linear independence for square matrices)
void *detHelper(void *param);                   // Multi-threading function
double determinant(double **matrix, int size);  // Calculates the determinant of a matrix

// Step 2: Orthogonalization using the Gram-Schmidt Process
double **orthogonalize(double **matrix, int row, int col);  // Creates an orthogonal basis for a set of vectors

// Step 3: Normalization
void *normHelper(void *param);                  // Multi-threading function
void normalizeVector(double *vector, int size); // Normalizes a vector (makes the magnitude equal to 1)


/* MAIN */

int main(int argc, char *argv[]) {
    // Argument error checking
    if (argc < 4) { // theoretical minimum is row=1, col=1, one element
        printf("ERROR: Too few arguments\n");
        return 0;
    }

    // First two arguments are the dimensions of the matrix
    int row = *argv[1] - '0', col = *argv[2] - '0';
    if (row < 1 || col < 1) {
        printf("ERROR: Invalid size %d, %d\n", row, col);
        return 0;
    }

    // Initialize matrix
    double **matrix = new double*[row];
    for (int i = 0; i < row; i++) {
        matrix[i] = new double[col];
    }

    // Read values from input to matrix
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (i * col + j + 3 > argc - 1) {
                printf("ERROR: Too few matrix values\n");
                return 0;
            }
            
            // Convert string of digits to number
            int num = 0;
            for  (char *digit = argv[i * col + j + 3]; *digit >= '0' && *digit <= '9'; digit++) {
                num *= 10;
                num += *digit - '0';
            }
            matrix[i][j] = (double)num;
        }
    }

    // Check if the columns of the matrix are linearly independent
    if (linearIndependence(matrix, row, col)) {
        printf("Matrix columns are linearly independent.\n");
    }
    else {
        printf("Matrix columns are not linearly independent.\n");
    }
    
    // Create an orthogonal basis (transposed)
    double **orthogonal = orthogonalize(matrix, row, col);
    
    // Set up threads for normalization
    pthread_t *threads = new pthread_t[col];
    normHelperArgs *args = new normHelperArgs[col];
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    // Convert the orthogonal basis to an orthonormal basis (transposed)
    for (int i = 0; i < col; i++) {
        // Set arguments
        args[i].v = orthogonal[i];
        args[i].size = row;

        // Create thread
        pthread_create(&threads[i], &attr, normHelper, (void *)&args[i]);
    }

    // Wait for threads to finish
    for (int i = 0; i < col; i++) {
        pthread_join(threads[i], NULL);
    }

    // Transpose the matrix back to its original state
    double **orthonormal = transpose(orthogonal, col, row);

    // Print the final result
    printf("Orthonormal Basis:\n");
    printMatrix(orthonormal, row, col, false);

    // Deallocate memory
    for (int i = 0; i < row; i++) {
        delete[] matrix[i];
        delete[] orthonormal[i];
    }
    for (int j = 0; j < col; j++) {
        delete[] orthogonal[j];
    }
    delete[] matrix;
    delete[] orthogonal;
    delete[] orthonormal;
    delete[] threads;
    delete[] args;

    return 0;
}


/* FUNCTION DEFINITIONS */

void printMatrix(double **matrix, int row, int col, bool includeZero) {

    // Checks if each column is entirely zero
    bool *nonZero = new bool[col];
    if (!includeZero) {
        for (int j = 0; j < col; j++) {
            nonZero[j] = false;
            for (int i = 0; i < row; i++) {
                if (matrix[i][j] != 0) {
                    nonZero[j] = true;
                    break;
                }
            }
        }
    }

    // Prints the matrix to the screen
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (includeZero || nonZero[j]) {
                if (matrix[i][j] >= 0) printf(" ");
                printf("%f ", matrix[i][j]);
            }
        }
        printf("\n");
    }

    delete[] nonZero;
}

double dotProduct(double *a, double *b, int size) {
    double prod = 0;
    for (int i = 0; i < size; i++)
        prod += a[i] * b[i];
    return prod;
}

double **transpose(double **matrix, int row, int col) {
    double **matrixT = new double*[col];
    for (int i = 0; i < col; i++) {
        matrixT[i] = new double[row];
        for (int j = 0; j < row; j++)
            matrixT[i][j] = matrix[j][i];
    }
    return matrixT;
}

void swapVectors(double *a, double *b, int size) {
    for (int i = 0, temp; i < size; i++) {
        temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

bool linearIndependence(double **matrix, int row, int col) {
    bool li = true;    // linear independence

    // If the matrix is square then linear independece can be checked with the determinant, which is easily multi-threaded
    // If the matrix is not square then the determinant is not defined and linear independence must be checked by row reduction, which is a sequential process

    // Square matrix
    if (row == col) {
        double det = 0;

        // Multithreading with a 2x2 matrix is entirely unecessary (even 3x3 is probably unhelpful)
        if (col < 3) {
            det = determinant(matrix, col);
        }
        else {

            // Setup threads
            pthread_t *threads = new pthread_t[col];
            pthread_attr_t attr;
            pthread_attr_init(&attr);

            // Arguments for thread calls
            detHelperArgs *args = new detHelperArgs[col];
            for (int i = 0; i < col; i++) {
                args[i].matrix = matrix;
                args[i].size = col;
                args[i].v = i;
                args[i].det = matrix[0][i];

                // Create threads
                pthread_create(&threads[i], &attr, detHelper, (void *)&args[i]);
            }

            // Combine results from threads
            int sign = 1;
            for (int i = 0; i < col; i++) {
                // Wait for thread to finish
                pthread_join(threads[i], NULL);

                // Add thread's result to the final determinant
                det += args[i].det * sign;
                sign *= -1;
            }

            // Deallocate memory
            delete[] threads;
            delete[] args;
        }

        // If the determinant is 0, the vectors are not linearly independent
        if (det == 0) li = false;
    }
    else if (col > row) {
        // Matrix with more columns than rows cannot be linearly independent
        li = false;
    }
    else {
        // Create a reduced row-echelon form matrix
        double **rrefMatrix = rref(matrix, row, col);
        if (rrefMatrix == nullptr) {
            printf("ERROR in rref(): returned nullptr\n");
            return false;
        }
        else {
            // Count the number of nonzero rows
            int nonzero = 0;
            for (int i = 0; i < row; i++) {
                double sum = 0;
                for (int j = 0; j < col; j++)
                    sum += rrefMatrix[i][j];
                if (sum != 0) nonzero++;
            }

            // If there are fewer nonzero rows than there are columns, then the vectors cannot be linearly independent
            if (nonzero < col) li = false;
        }

        // Deallocate memory
        for (int i = 0; i < row; i++)
            delete[] rrefMatrix[i];
        delete[] rrefMatrix;
    }

    return li;
}

double **rref(double **matrix, int row, int col) {
    // Based on the psuedocode for the REMEF algorithm from http://linear.ups.edu/jsmath/0210/fcla-jsmath-2.10li18.html

    // Create a new matrix to hold the rref form
    double **rrefMatrix = new double*[row];
    for (int i = 0; i < row; i++) {
        rrefMatrix[i] = new double[col];
        for (int j = 0; j < col; j++) {
            rrefMatrix[i][j] = matrix[i][j];
        }
    }

    for (int j = 0, r = -1, i; j < col; j++, i = r + 1) {

        // Find next nonzero row
        while (i < row && rrefMatrix[i][j] == 0) { i++; }

        if (i < row) {
            // Swap rows i and r
            if (i != ++r) swapVectors(rrefMatrix[i], rrefMatrix[r], col);
            
            // Scale row r so that its leading entry is 1
            double scalar = rrefMatrix[r][j];
            for (int l = j; l < col; l++) {
                rrefMatrix[r][l] /= scalar;
            }

            // Zero out the other entries in column j
            for (int k = 0; k < row; k++) {
                if (k != r) {
                    // Scalar such that the third row operation makes column j entries zero 
                    scalar = 0;
                    if (rrefMatrix[r][j] != 0)  // Don't divide by zero
                        scalar = -rrefMatrix[k][j] / rrefMatrix[r][j];

                    for (int l = j; l < col; l++) {
                        rrefMatrix[k][l] += rrefMatrix[r][l] * scalar;
                    }
                }
            }
        }
    }

    return rrefMatrix;
}

void *detHelper(void *param) {
    detHelperArgs *args = (detHelperArgs *)param;

    // Create a submatrix
    double **subMatrix = new double*[args->size - 1];
    for (int i = 0; i < args->size - 1; i++) {
        subMatrix[i] = new double[args->size - 1];
    }

    // Fill submatrix
    for (int i = 1, k; i < args->size; i++, k = 0) {
        for (int j = 0; j < args->size; j++) {
            if (j != args->v) {
                subMatrix[i - 1][k++] = args->matrix[i][j];
            }
        }
    }

    // Calculate determinant of submatrix
    args->det = args->det * determinant(subMatrix, args->size - 1);

    // Deallocate memory
    for (int i = 0; i < args->size - 1; i++)
        delete[] subMatrix[i];
    delete[] subMatrix;

    pthread_exit(0);
}

double determinant(double **matrix, int size) {
    // Base cases
    if (size == 1) return matrix[0][0];
    if (size == 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

    // Need a (size - 1)x(size - 1) matrix for each column
    double ***subMatrices = new double**[size];
    for (int i = 0; i < size; i++) {
        subMatrices[i] = new double*[size - 1];
        for (int j = 0; j < size - 1; j++) {
            subMatrices[i][j] = new double[size - 1];
        }
    }

    // Fill submatrices
    for (int i = 0; i < size; i++) {
        for (int row = 1; row < size; row++) {
            int j = 0;
            for (int col = 0; col < size; col++) {
                if (col == i) continue; // Per the cofactor algorithm
                if (j == size - 1 || col == size) {
                    return 0;
                }
                subMatrices[i][row - 1][j++] = matrix[row][col];
            }
        }
    }

    // Calculate determinant based on determinants of submatrices
    int sign = 1;
    double det = 0;
    for (int i = 0; i < size; i++){
        det += matrix[0][i] * determinant(subMatrices[i], size - 1) * sign;
        sign *= -1;
    }

    // Deallocate memory
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size - 1; j++)
            delete[] subMatrices[i][j];
        delete[] subMatrices[i];
    }
    delete[] subMatrices;

    return det;
}

double **orthogonalize(double **matrix, int row, int col) {
    // Orthogonal matrix (transposed)
    double **orthogonal = new double*[col];
    for (int i = 0; i < col; i++) {
        orthogonal[i] = new double[row];
    }

    // This step deals with columns instead of rows, so it's easier to work with the tranpose
    double **matrixT = transpose(matrix, row, col);
    double *dotProducts = new double[col];

    for (int i = 0; i < col; i++) {

        // Get a vector of the matrix
        for (int j = 0; j < row; j++) {
            orthogonal[i][j] = matrixT[i][j];
        }

        
        // Subtract the projections of the current vector onto previous vectors (to make them orthogonal)
        for (int k = 0; k < i; k++) {
            if (abs(dotProducts[k]) > 1e-9) {   // Weird rounding error was causing an equality check to return false even when it was zero
                double projScalar = dotProduct(matrixT[i], orthogonal[k], row) / dotProducts[k];
                for (int j = 0; j < row; j++) {
                    orthogonal[i][j] -= projScalar * orthogonal[k][j];
                }
            }
        }

        // Save dot products to avoid unecessary computation later
        dotProducts[i] = dotProduct(orthogonal[i], orthogonal[i], row);
    }

    // Deallocate memory
    for (int i = 0; i < col; i++) {
        delete[] matrixT[i];
    }
    delete[] matrixT;
    delete[] dotProducts;

    // Return the orthogonalized matrix (transposed from the original)
    return orthogonal;
}

void *normHelper(void *param) {
    // Normalize the single given vector
    normHelperArgs *args = (normHelperArgs *)param;
    normalizeVector(args->v, args->size);
    pthread_exit(0);
}

void normalizeVector(double *vector, int size) {
    // Divide each element by the vector's magnitude
    double mag = sqrt(dotProduct(vector, vector, size));
    for (int i = 0; i < size; i++) {
        if (abs(mag) > 1e-9) vector[i] /= mag;   // Same rounding error as above
        else vector[i] = 0;
    }
}