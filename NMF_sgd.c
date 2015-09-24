#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<cuda_runtime.h> 
#include<cusparse.h>
#include<cublas_v2.h>

#define MAX_STRING 100
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
typedef float real; 
const real zero = 0.0;
const real one = 1.0;
const real negOne = -1.0;

/* define variables */
cusparseHandle_t handle_sparse = 0;
cusparseMatDescr_t descr_sparse = 0;
cublasHandle_t handle_blas = 0;
cudaError_t cudaStat;       //for cuda errors
int m, n=2, k;                //V=WH, V:m*k, W:m*n, H:n*k
int lineNumber = -1;        //line number(positive value) of V
FILE *file;                 //file handle
char _str[MAX_STRING];      //a black hole for string reading
int tmpRow, tmpCol;         //tmp variables for reading sparse matrix index
real tmpVal;                //tmp variable for reading sparse matrix value  
int *VRowIndexHost;         //row index of V in host
int *VColIndexHost;         //column index of V in host
real *VValHost;             //value of V in host
real *WValHost, *HValHost;  //value of W,H in host 
int *VRowCoo;               //row index of V in GPU in COO format, for reading data only
int *VRow;                  //row index of V in GPU
int *VCol;                  //col index of V in GPU
real *V;                    //V in GPU
real *W, *H;                //W,H in GPU

char filename[100];         //the file of V, store as sparse matrix 
int gpuid = 0;              //GPU to use
real lrate = 0.05;          //learning rate of sub problem
int maxiterMain = 500;      //max iter number of main problem 
int maxiterSub = 100;       //max iter number of sub problem 

/* a macro for free memory*/
#define CLEANUP(s)                                  \
do {                                                \
    printf ("%s\n", s);                             \
    if (WValHost) free(WValHost);                   \
    if (HValHost) free(HValHost);                   \
    if (V) cudaFree(V);                             \
    if (VRow) cudaFree(VRow);                       \
    if (VCol) cudaFree(VCol);                       \
    if (W) cudaFree(W);                             \
    if (H) cudaFree(H);                             \
    cusparseDestroy(handle_sparse);                 \
    cusparseDestroyMatDescr(descr_sparse);          \
	cublasDestroy(handle_blas);                     \
    cudaDeviceReset();                              \
    fflush (stdout);                                \
} while (0)

/* random init a array data of size p */
void randomInit(real *data, int p){
    for (int i = 0; i < p; ++i)
        data[i] = rand() / (real)RAND_MAX;
}

/* print a matrix of size row*col */
void outPutMatrix(int row, int col, real *A)
{
	int i, j;
    for(i = 0; i < row; i++){
        for(j = 0; j < col; j++)
            printf("%10.4f ", A[IDX2C(i,j,row)]);
        printf("\n");
    }
}

/* init variables */
void initVaribles(){

    /* allocate memory and initial */
    if((file = fopen(filename, "r")) == NULL){
        printf("File %s not found!\n", filename);
        exit(1);
    }
    while(fgets(_str, sizeof(_str), file))
        lineNumber++;
    fclose(file);
    printf("Line number(positive value) of V: %d\n", lineNumber);
    VRowIndexHost = (int *)malloc(lineNumber*sizeof(int));
    VColIndexHost = (int *)malloc(lineNumber*sizeof(int));
    VValHost = (real *)malloc(lineNumber*sizeof(real));
    WValHost = (real *)malloc(m*n*sizeof(real));
    HValHost = (real *)malloc(n*k*sizeof(real));
    if((!VRowIndexHost) || (!VColIndexHost) || (!VValHost) || !(WValHost) || !(HValHost)){
        CLEANUP("Host malloc failed (matrix)");
        exit(1);
    }
    file = fopen(filename, "r");
    int i = 0;
    fscanf(file, "%d %d", &m, &k);
    while(fscanf(file, "%d %d %f", &tmpRow, &tmpCol, &tmpVal) != EOF){
        VRowIndexHost[i] = tmpRow;
        VColIndexHost[i] = tmpCol;
        VValHost[i] = tmpVal;
        i++;
    }
    fclose(file);
    randomInit(WValHost, m*n);
    randomInit(HValHost, n*k);

    printf("Matrix shape of m n k: %d %d %d\n", m, n, k);
    /*
    printf("W:\n");
    outPutMatrix(m, n, WValHost);
    printf("H:\n");
    outPutMatrix(n, k, HValHost);
    */
}

/* shipping data to GPU */
void shipping(){
    cudaStat = cudaSetDevice(gpuid);
    if(cudaStat != cudaSuccess){
        CLEANUP("Device not found, check your gpuid!");
        exit(1);
    }
    cudaMalloc((void**)&VRowCoo, lineNumber*sizeof(int)); 
    cudaMalloc((void**)&VCol, lineNumber*sizeof(int)); 
    cudaMalloc((void**)&V, lineNumber*sizeof(real)); 
    cudaMalloc((void**)&W, m*n*sizeof(real)); 
    cudaMalloc((void**)&H, n*k*sizeof(real)); 

    cudaMemcpy(VRowCoo, VRowIndexHost, (size_t)(lineNumber*sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(VCol, VColIndexHost, (size_t)(lineNumber*sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(V, VValHost, (size_t)(lineNumber*sizeof(real)), cudaMemcpyHostToDevice);
    cudaMemcpy(W, WValHost, (size_t)(m*n*sizeof(real)), cudaMemcpyHostToDevice);
    cudaMemcpy(H, HValHost, (size_t)(n*k*sizeof(real)), cudaMemcpyHostToDevice);

    /* setup cusparse and cublas library */
    cusparseCreate(&handle_sparse); 
    cusparseCreateMatDescr(&descr_sparse);
    cusparseSetMatType(descr_sparse,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_sparse,CUSPARSE_INDEX_BASE_ZERO);
    cublasCreate(&handle_blas);

    /* convert V from COO 2 CSR format */
    cudaMalloc((void**)&VRow,(m+1)*sizeof(int));
    cusparseXcoo2csr(handle_sparse, VRowCoo, lineNumber, m, VRow, CUSPARSE_INDEX_BASE_ZERO);

    //slowTest
    /*
    real *Vdense, *VdenseHost;
    cudaMalloc((void**)&Vdense, m*k*sizeof(real)); 
    cusparseScsr2dense(handle_sparse, m, k, descr_sparse, V, VRow, VCol, Vdense, m);
    VdenseHost = (real *)malloc(m*k*sizeof(real));
    cudaMemcpy(VdenseHost, Vdense, (size_t)(m*k*sizeof(real)), cudaMemcpyDeviceToHost);
    printf("V:\n");
    outPutMatrix(m, k, VdenseHost);
    */

    /* free some useless variables */
    if (VValHost) free(VValHost);                   
    if (VRowIndexHost) free(VRowIndexHost);        
    if (VColIndexHost) free(VColIndexHost);       
    if (VRowCoo) cudaFree(VRowCoo);        
}

/* shipping back to host */
void backHost(){
    cudaMemcpy(WValHost, W, (size_t)(m*n*sizeof(real)), cudaMemcpyDeviceToHost); 
    cudaMemcpy(HValHost, H, (size_t)(n*k*sizeof(real)), cudaMemcpyDeviceToHost); 
}

real subprob(real *V, cusparseOperation_t transV, int rowV, int colV, real *W, real *H, 
             int mm, int nn, int kk, real lrate, int maxiter2, int *realIter, real tol){
    real *VtW, *WtV, *WtW, *grad, curGrad = 0;
    cudaMalloc((void**)&VtW, kk*nn*sizeof(real)); 
    cudaMalloc((void**)&WtV, nn*kk*sizeof(real)); 
    cudaMalloc((void**)&WtW, nn*nn*sizeof(real)); 
    cudaMalloc((void**)&grad, nn*kk*sizeof(real)); 

    int iter = 0;
	for(int iter = 1; iter <= maxiter2; iter++){
        //VtW = V'*W
        cusparseScsrmm(handle_sparse, transV, rowV, nn, colV, lineNumber, &one, descr_sparse, V, VRow, VCol, W, mm, &zero, VtW, kk);
        //WtV = (VtW)'
        cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, nn, kk, &one, VtW, kk, &zero, WtV, nn, WtV, nn);
        //WtW = W'*W;
        cublasSgemm(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, nn, nn, mm, &one, W, mm, W, mm, &zero, WtW, nn);
        //grad = WtW*H - WtV;
        cudaMemcpy(grad, WtV, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);
        cublasSgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, nn, kk, nn, &one, WtW, nn, H, nn, &negOne, grad, nn);
        //H = H - lrate * grad
        real lrateTmp = -lrate;
        cublasSgeam(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, nn, kk, &one, H, nn, &lrateTmp, grad, nn, H, nn);
        //||grad||2
        cublasSnrm2(handle_blas, nn*kk, grad, 1, &curGrad);
        //printf("\tSub-Iter %d, norm of grad: %10.6f\n", iter, curGrad);
        if(curGrad < tol)
            break;
    }
	*realIter = iter;

    cudaFree(VtW);
    cudaFree(WtV);
    cudaFree(WtW);
    cudaFree(grad);
    return curGrad;
}

/* NMF */
void NMF(real *V, int *VRow, int *VCol, real lrate, int maxiter, int maxiter2, real *W, real *H){
    real *Wt, *Ht; //Wt, Ht
    real tolH = 0.001, tolW = 0.001, tol = 0.0001;
    real grad1 = 0, grad2 = 0, curGrad = 0, initGrad = 0;

    int nochange = 0;
    cudaMalloc((void**)&Wt, m*n*sizeof(real)); 
    cudaMalloc((void**)&Ht, n*k*sizeof(real)); 

	for(int iter = 1; iter <= maxiter; iter++){
        //update H, gradH = WtWH-(VtW)t
        int iterH = 0;
        grad1 = subprob(V, CUSPARSE_OPERATION_TRANSPOSE, m, k, W, H, m, n, k, lrate, maxiter2, &iterH, tolH);
		if (iterH == 1 && tolH > 0.000001)
			tolH = 0.1 * tolH;
        cudaMemcpy(HValHost, H, (size_t)(n*k*sizeof(real)), cudaMemcpyDeviceToHost); 
        for(int i = 0; i < n*k; i++)
            if(HValHost[i] < 0)
                HValHost[i] = 0;
        cudaMemcpy(H, HValHost, (size_t)(n*k*sizeof(real)), cudaMemcpyHostToDevice);

        //update W, Vt = HtWt, then Wt is the same as H before
        cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &one, W, m, &zero, Wt, n, Wt, n); //Wt
        cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, &one, H, n, &zero, Ht, k, Ht, k); //Ht
        int iterW = 0;
        grad2 = subprob(V, CUSPARSE_OPERATION_NON_TRANSPOSE, m, k, Ht, Wt, k, n, m, lrate, maxiter2, &iterW, tolW);
		if (iterW == 1 && tolW > 0.000001)
			tolW = 0.1 * tolW;
        cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &one, Wt, n, &zero, W, m, W, m); // W = Wt'
        cudaMemcpy(WValHost, W, (size_t)(m*n*sizeof(real)), cudaMemcpyDeviceToHost); 
        for(int i = 0; i < m*n; i++)
            if(WValHost[i] < 0)
                WValHost[i] = 0;
        cudaMemcpy(W, WValHost, (size_t)(m*n*sizeof(real)), cudaMemcpyHostToDevice);

        //stop when grad < tol* initGrad
        curGrad = grad1 + grad2;
        //printf("Iter %d, norm of grad: %10.6f\n", iter, curGrad);
        if(iter == 1)
            initGrad = curGrad;
        if(curGrad < tol*initGrad)
            break;
    }

    cudaFree(Wt);
    cudaFree(Ht);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv){
	int i;
	if (argc == 1) {
		printf("NMF: Non-negative Matrix Factorization\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse data from <file> to train the model;\n");
		printf("\t-tlrate <float>\n");
		printf("\t\tlearning rate; default is 0.05\n");
		printf("\t-tfactor <int>\n");
		printf("\t\tfactor number; default is 2\n");
		printf("\t-titerMain <int>\n");
		printf("\t\tmax iter number of main loop; default is 500\n");
		printf("\t-titerSub <int>\n");
		printf("\t\tmax iter number of sub problem; default is 100\n");
		printf("\t-tgpuid <int>\n");
		printf("\t\twhich gpu to use; default is 0\n");
		printf("\nExamples:\n");
		printf("./NMF_sgd -train test.txt -lrate 0.05 -factor 3 -iterMain 500 -iterSub 100 -gpuid 0\n\n");
		return 0;
	}
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(filename, argv[i + 1]);
	if ((i = ArgPos((char *)"-lrate", argc, argv)) > 0) lrate = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-factor", argc, argv)) > 0) n = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iterMain", argc, argv)) > 0) maxiterMain = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iterSub", argc, argv)) > 0) maxiterSub = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-gpuid", argc, argv)) > 0) gpuid = atoi(argv[i + 1]);

    initVaribles();
    shipping();
    NMF(V, VRow, VCol, lrate, maxiterMain, maxiterSub, W, H);

    //get result
    backHost();
    //save result
    FILE *f = fopen("W.txt", "w");
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)
            fprintf(f, "%.4f ", WValHost[IDX2C(i,j,m)]);
        fprintf(f, "\n");
    }
    fclose(f);
    f = fopen("H.txt", "w");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < k; j++)
            fprintf(f, "%.4f ", HValHost[IDX2C(i,j,n)]);
        fprintf(f, "\n");
    }
    fclose(f);

    /*
    printf("Result:\n");
    printf("W:\n");
    outPutMatrix(m, n, WValHost);
    printf("H:\n");
    outPutMatrix(n, k, HValHost);
    */

    //slowTest
    /*
    real *Vdense, *VdenseHost;
    cudaMalloc((void**)&Vdense, m*k*sizeof(real)); 
    cublasSgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &one, W, m, H, n, &zero, Vdense, m);
    cudaMemcpy(VdenseHost, Vdense, (size_t)(m*k*sizeof(real)), cudaMemcpyDeviceToHost); 
    printf("WH:\n");
    outPutMatrix(m, k, VdenseHost);
    */

    CLEANUP("done.");
}
