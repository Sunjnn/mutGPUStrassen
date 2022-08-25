// Define matrix size.
// Matrix A has the size of M times K.
// Matrix B has the size of K times N.
// Matrix C has the size of M times N.
#ifndef MATRIX_SIZE
#define MATRIX_SIZE
#define M 8192
#define K 8192
#define N 8192
#endif


// Define sub matrix size.
#ifndef BLOCK_MATRIX_SIZE
#define BLOCK_MATRIX_SIZE
#define BLOCK_M 1024
#define BLOCK_K 1024
#define BLOCK_N 1024
#endif


#ifndef __EXIT__
#define __EXIT__
#define EXIT() {                    \
    printf("Enter to exit:");      \
    getchar();                      \
}
#endif
