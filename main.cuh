// Define matrix size.
// Matrix A has the size of M times K.
// Matrix B has the size of K times N.
// Matrix C has the size of M times N.
#ifndef MATRIX_SIZE
#define MATRIX_SIZE
#define M 32768
#define K M
#define N M
#endif


// Define sub matrix size.
#ifndef BLOCK_MATRIX_SIZE
#define BLOCK_MATRIX_SIZE
#define BLOCK_M 8192
#define BLOCK_K BLOCK_M
#define BLOCK_N BLOCK_M
#endif


#ifndef __EXIT__
#define __EXIT__
#define EXIT() {                    \
    printf("Enter to exit:");      \
    getchar();                      \
}
#endif
