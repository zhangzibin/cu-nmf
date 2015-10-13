#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<cusparse.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>

typedef float real;
#define MAX_STRING 100
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define bool int
#define false 0
#define true 1
#define max(a,b)(a>b)?a:b
const real zero = 0.0;
const real one = 1.0;
const real negOne = -1.0;

cusparseHandle_t handle_sparse = 0;
cusparseMatDescr_t descr_sparse = 0;
cublasHandle_t handle_blas = 0;
cudaError_t cudaStat;       //for cuda errors
dim3 threadsPerBlock(16);

/* define variables */
char filename[100];         //the file of V, store as sparse matrix
int gpuid = 0;              //GPU to use
int m, n = 2, k;              //V=WH, V:m*k, W:m*n, H:n*k
real tol = 0.001;
real timelimit = 100;
int maxiter = 10;
int lineNumber = -1;        //line number(positive value) of V
FILE *file;                 //file handle
char _str[MAX_STRING];      //a black hole for string reading
int tmpRow, tmpCol;         //tmp variables for reading sparse matrix index
real tmpVal;                //tmp variable for reading sparse matrix value

int *VRowIndexHost = 0;     //row index of V in host
int *VColIndexHost = 0;     //column index of V in host
real *VHost = 0;            //value of V in host
real *WHost = 0;            //value of W in host
real *HHost = 0;            //value of H in host

int *VRowCoo = 0;           //row index of V in GPU in COO format, for reading data only
int *VRow = 0;              //row index of V in GPU
int *VCol = 0;              //col index of V in GPU
real *V = 0;                //V in GPU
real *W = 0, *H = 0;        //W,H in GPU

/* a macro for free memory*/
#define CLEANUP(s)                                  \
do {                                                \
    printf ("%s\n", s);                             \
    if (WHost) free(WHost);                         \
    if (HHost) free(HHost);                         \
    if (VRow) cudaFree(VRow);                       \
    if (VCol) cudaFree(VCol);                       \
    if (V) cudaFree(V);                             \
    if (W) cudaFree(W);                             \
    if (H) cudaFree(H);                             \
    cusparseDestroy(handle_sparse);                 \
    cusparseDestroyMatDescr(descr_sparse);          \
    cublasDestroy(handle_blas);                     \
    cudaDeviceReset();                              \
    fflush (stdout);                                \
} while (0)

void randomInit(real *data, int p){
    int i = 0;
    for (; i < p; ++i)
        data[i] = rand() / (real)RAND_MAX;
}

/* print a matrix of size row*col */
void outPutMatrix(int row, int col, real *A){
    int i, j;
    for(i = 0; i < row; i++){
        for(j = 0; j < col; j++)
            printf("%10.4f ", A[IDX2C(i,j,row)]);
        printf("\n");
    }
}

//clip negative value
__global__ void clipNegative(real *A, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && A[i] < 0)
        A[i] = 0;
}

//projgrad = norm(grad(grad < 0 | H >0)), let tmpvec is the useful values
__global__ void getUsefulGrad(real *grad, real *H, real *tmpvec, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        if(grad[i] < 0 || H[i] < 0)
            tmpvec[i] = grad[i];
}

//projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
__global__ void getUsefulGrad2(real *grad, real *H, real *tmpvec, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        if(grad[i] < 0 || H[i] > 0)
            tmpvec[i] = grad[i];
}

void initVaribles(){
    srand((unsigned)time(NULL));

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
    VHost = (real *)malloc(lineNumber*sizeof(real));

    file = fopen(filename, "r");
    fscanf(file, "%d %d", &m, &k);
    printf("Matrix shape of m n k: %d %d %d\n", m, n, k);

    WHost = (real *)malloc(m*n*sizeof(real));
    randomInit(WHost, m*n);
    HHost = (real *)malloc(n*k*sizeof(real));
    randomInit(HHost, n*k);

    int i = 0;
    while(fscanf(file, "%d %d %f", &tmpRow, &tmpCol, &tmpVal) != EOF){
        *(VRowIndexHost+i) = tmpRow;
        *(VColIndexHost+i) = tmpCol;
        *(VHost+i) = tmpVal;
        i++;
    }
    fclose(file);

    cudaStat = cudaSetDevice(gpuid);
    if(cudaStat != cudaSuccess){
        CLEANUP("Device not found, check your gpuid!");
        exit(1);
    }
    /* setup cusparse and cublas library */
    cusparseCreate(&handle_sparse);
    cusparseCreateMatDescr(&descr_sparse);
    cusparseSetMatType(descr_sparse,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_sparse,CUSPARSE_INDEX_BASE_ZERO);
    cublasCreate(&handle_blas);
}

/* shipping data to GPU */
void shipping(){
    cudaMalloc((void**)&VRowCoo, lineNumber*sizeof(int));
    cudaMalloc((void**)&VCol, lineNumber*sizeof(int));
    cudaMalloc((void**)&V, lineNumber*sizeof(real));
    cudaMalloc((void**)&W, m*n*sizeof(real));
    cudaMalloc((void**)&H, n*k*sizeof(real));

    cudaMemcpy(VRowCoo, VRowIndexHost, (size_t)(lineNumber*sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(VCol, VColIndexHost, (size_t)(lineNumber*sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(V, VHost, (size_t)(lineNumber*sizeof(real)), cudaMemcpyHostToDevice);
    cudaMemcpy(W, WHost, (size_t)(m*n*sizeof(real)), cudaMemcpyHostToDevice);
    cudaMemcpy(H, HHost, (size_t)(n*k*sizeof(real)), cudaMemcpyHostToDevice);

    /* convert V from COO 2 CSR format */
    cudaMalloc((void**)&VRow,(m+1)*sizeof(int));
    cusparseXcoo2csr(handle_sparse, VRowCoo, lineNumber, m, VRow, CUSPARSE_INDEX_BASE_ZERO);

    //print V for test
    real *Vdense, *VdenseHost;
    cudaMalloc((void**)&Vdense, m*k*sizeof(real));
    cusparseScsr2dense(handle_sparse, m, k, descr_sparse, V, VRow, VCol, Vdense, m);
    VdenseHost = (real *)malloc(m*k*sizeof(real));
    cudaMemcpy(VdenseHost, Vdense, (size_t)(m*k*sizeof(real)), cudaMemcpyDeviceToHost);
    printf("V:\n");
    outPutMatrix(m, k, VdenseHost);

    /* free some useless variables */
    if (VHost) free(VHost);
    if (VRowIndexHost) free(VRowIndexHost);
    if (VColIndexHost) free(VColIndexHost);
    if (VRowCoo) cudaFree(VRowCoo);
}

void subprob(real *V, cusparseOperation_t transV, int rowV, int colV, real *W, real *Hinit,
            int mm, int nn, int kk, real tol, int maxiter, real *H, real *grad, int *ite){
    //H = Hinit
    cudaMemcpy(H, Hinit, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);

    //WtV = W'*V;
    real *WtV = 0, *VtW = 0;
    cudaMalloc((void**)&WtV, nn*kk*sizeof(real));
    cudaMalloc((void**)&VtW, kk*nn*sizeof(real));
    cusparseScsrmm(handle_sparse, transV, rowV, nn, colV, lineNumber, &one, descr_sparse, V, VRow, VCol, W, mm, &zero, VtW, kk);//VtW = V'*W
    cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, nn, kk, &one, VtW, kk, &zero, WtV, nn, WtV, nn);//WtV = (VtW)'
    cudaFree(VtW);

    //WtW = W'*W;
    real *WtW = 0;
    cudaMalloc((void**)&WtW, m*m*sizeof(real));
    cublasSgemm(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, nn, nn, mm, &one, W, mm, W, mm, &zero, WtW, nn);

    real alpha = 1;
    real beta = 0.1;

    real *Hn = 0;
    cudaMalloc((void**)&Hn, nn*kk*sizeof(real));
    cudaMemcpy(Hn, H, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);

    real *d = 0;
    cudaMalloc((void**)&d, nn*kk*sizeof(real));
    cudaMemset(d, 0, nn*kk*sizeof(real));

    real *WtWd = 0;
    cudaMalloc((void**)&WtWd, nn*kk*sizeof(real));
    cudaMemset(WtWd, 0, nn*kk*sizeof(real));

    real *Hp = 0;
    cudaMalloc((void**)&Hp, nn*kk*sizeof(real));
    cudaMemset(Hp, 0, nn*kk*sizeof(real));

    real *Hnpp = 0;
    cudaMalloc((void**)&Hnpp, nn*kk*sizeof(real));
    cudaMemset(Hnpp, 0, nn*kk*sizeof(real));

    real *tmpvec = 0;
    cudaMalloc((void**)&tmpvec, nn*kk*sizeof(real));

    int iter = 0;
    for(iter = 1; iter <= maxiter; iter++){
        //grad = WtW*H - WtV;
        cudaMemcpy(grad, WtV, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);//grad = WtV (tmp step)
        cublasSgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, nn, kk, nn, &one, WtW, nn, H, nn, &negOne, grad, nn);//grad = WtW*H - WtV;
        //projgrad = norm(grad(grad < 0 | H >0))
        cudaMemset(tmpvec, 0, nn*kk*sizeof(real));
        dim3 num1(nn*kk / threadsPerBlock.x + 1);
        getUsefulGrad<<<num1, threadsPerBlock>>>(grad, H, tmpvec, nn*kk);
        real projgrad = 0;
        cublasSnrm2(handle_blas, nn*kk, tmpvec, 1, &projgrad);
        //printf("projgrad %f\n", projgrad);
        if (projgrad < tol)
            break;

        int inner_iter = 1;
        for(; inner_iter <= 20; inner_iter++){
            //Hn = max(H - alpha*grad, 0); d = Hn-H;
            cudaMemcpy(Hn, H, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);
            real nalpha = -alpha;
            cublasSaxpy(handle_blas, nn*kk, &nalpha, grad, 1, Hn, 1);
            dim3 num2(nn*kk / threadsPerBlock.x + 1);
            clipNegative<<<num2, threadsPerBlock>>>(Hn, nn*kk);
            cudaMemcpy(d, Hn, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);
            cublasSaxpy(handle_blas, nn*kk, &negOne, H, 1, d, 1);

            //gradd=sum(sum(grad.*d)); dQd = sum(sum((WtW*d).*d));
            real gradd = 0, dQd = 0;
            cublasSdot(handle_blas, nn*kk, grad, 1, d, 1, &gradd);
            cublasSgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, nn, kk, nn ,&one, WtW, nn, d, nn, &zero, WtWd, nn);
            cublasSdot(handle_blas, nn*kk, WtWd, 1, d, 1, &dQd);
            bool suff_decr = 0.99*gradd + 0.5*dQd < 0;
            bool decr_alpha = true;
            if (inner_iter == 1){
                decr_alpha = ~suff_decr;
                cudaMemcpy(Hp, H, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);
            }
            if(decr_alpha){
                if(suff_decr){
                    cudaMemcpy(H, Hn, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);
                    break;
                }
                else
                    alpha = alpha * beta;
            }
            else{
                cudaMemcpy(Hnpp, Hn, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);
                cublasSaxpy(handle_blas, nn*kk, &negOne, Hp, 1, Hnpp, 1);
                real test = 0;
                cublasSnrm2(handle_blas, nn*kk, Hnpp, 1, &test);
                if(~suff_decr || test == 0){
                    cudaMemcpy(H, Hp, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);
                    break;
                }
                else{
                    alpha = alpha/beta;
                    cudaMemcpy(Hp, Hn, nn*kk*sizeof(real), cudaMemcpyDeviceToDevice);
                }
            }
        }
    }
    *ite = iter;
}

void NMF(){
    clock_t initt = time(NULL);

    //gradW = W*(H*H') - V*H';
    real *HHt = 0, *gradW = 0, *gradWt = 0,*VHt = 0;
    cudaMalloc((void**)&HHt, n*n*sizeof(real));
    cudaMalloc((void**)&gradW, m*n*sizeof(real));
    cudaMalloc((void**)&gradWt, m*n*sizeof(real));
    cudaMalloc((void**)&VHt, m*n*sizeof(real));
    cusparseScsrmm2(handle_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
            m, n, k,lineNumber, &one, descr_sparse, V, VRow, VCol, H, n, &zero, VHt, m); //VHt = V*H'
    cudaMemcpy(gradW, VHt, m*n*sizeof(real), cudaMemcpyDeviceToDevice); //gradW = VHt (tmp step)
    cublasSgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &one, H, n, H, n, &zero, HHt, n); //HHt = H*H'
    cublasSgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &one, W, m, HHt, n, &negOne, gradW, m); //gradW = W*(H*H') - V*H'

    //gradH = (W'*W)*H - W'*V;
    real *WtW = 0, *gradH = 0, *WtV = 0, *VtW = 0;
    cudaMalloc((void**)&WtW, m*m*sizeof(real));
    cudaMalloc((void**)&gradH, n*k*sizeof(real));
    cudaMalloc((void**)&WtV, n*k*sizeof(real));
    cudaMalloc((void**)&VtW, k*n*sizeof(real));
    cusparseScsrmm(handle_sparse, CUSPARSE_OPERATION_TRANSPOSE, k, n, m,
            lineNumber, &one, descr_sparse, V, VRow, VCol, W, m, &zero, VtW, k);//VtW = V'*W
    cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, k, &one, VtW, k, &zero, WtV, n, WtV, n);//WtV = (VtW)'
    cublasSgemm(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &one, W, m, W, m, &zero, WtW, n);//WtW = W'*W
    cudaMemcpy(gradH, WtV, n*k*sizeof(real), cudaMemcpyDeviceToDevice);//gradH = WtV (tmp step)
    cublasSgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, WtW, n, H, n, &negOne, gradH, n);//gradH = WtW*H - WtV;

    //initgrad = norm(gradW) + norm(gradH);
    real initgrad = 0, tmpgrad = 0;
    cublasSdot(handle_blas, m*n, gradW, 1, gradW, 1, &initgrad);
    cublasSdot(handle_blas, n*k, gradH, 1, gradH, 1, &tmpgrad);
    initgrad += tmpgrad;
    initgrad = sqrt(initgrad);
    printf("Init gradient norm %f\n", initgrad);
    real tolW = initgrad*max(0.001,tol);
    real tolH = tolW;

    real *tmpvec, *tmpvec2; //W, H
    cudaMalloc((void**)&tmpvec, m*n*sizeof(real));
    cudaMemset(tmpvec, 0, m*n*sizeof(real));
    cudaMalloc((void**)&tmpvec2, n*k*sizeof(real));
    cudaMemset(tmpvec2, 0, n*k*sizeof(real));

    real *Wt, *Ht; //Wt, Ht
    cudaMalloc((void**)&Wt, m*n*sizeof(real));
    cudaMalloc((void**)&Ht, n*k*sizeof(real));

    int iter = 0;
    real projnorm = 0, tmpnorm = 0, lastnorm = 0;
    for(iter = 1; iter <= maxiter; iter++){
        //stopping condition
        //projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
        dim3 num3(m*n / threadsPerBlock.x + 1);
        getUsefulGrad2<<<num3, threadsPerBlock>>>(gradW, W, tmpvec, m*n);
        cublasSdot(handle_blas, m*n, tmpvec, 1, tmpvec, 1, &projnorm);
        dim3 num4(n*k / threadsPerBlock.x + 1);
        getUsefulGrad2<<<num4, threadsPerBlock>>>(gradH, H, tmpvec2, n*k);
        cublasSdot(handle_blas, n*k, tmpvec2, 1, tmpvec2, 1, &tmpnorm);
        projnorm += tmpnorm;
        projnorm = sqrt(projnorm);
        printf("Iter %d, projnorm %f\n", iter, projnorm);
        if(iter != 1 && projnorm == lastnorm)
            break;
        if(projnorm < tol*initgrad || time(NULL)-initt > timelimit)
            break;
        lastnorm = projnorm;

        //update W, Vt = HtWt, then Wt is the same as H before
        cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &one, W, m, &zero, Wt, n, Wt, n); //Wt
        cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, &one, H, n, &zero, Ht, k, Ht, k); //Ht
        int iterW = 0;
        subprob(V, CUSPARSE_OPERATION_NON_TRANSPOSE, m, k, Ht, Wt, k, n, m, tolW, 1000, W, gradW, &iterW);
        if(iterW == 1 && tolW > 0.000001)
            tolW = 0.1 * tolW;
        //W = W' , gradW = gradW'
        cudaMemcpy(Wt, W, (size_t)(m*n*sizeof(real)), cudaMemcpyDeviceToDevice);
        cudaMemcpy(gradWt, gradW, (size_t)(m*n*sizeof(real)), cudaMemcpyDeviceToDevice);
        cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &one, Wt, n, &zero, W, m, W, m);
        cublasSgeam(handle_blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &one, gradWt, n, &zero, gradW, m, gradW, m);

        //update H
        int iterH = 0;
        subprob(V, CUSPARSE_OPERATION_TRANSPOSE, m, k, W, H, m, n, k, tolH, 1000, H, gradH, &iterH);
        if(iterH == 1 && tolH > 0.000001)
            tolH = 0.1 * tolH;
        //printf("HH, %d\n", iterH);
    }
}

/* shipping back to host */
void backHost(){
    cudaMemcpy(WHost, W, (size_t)(m*n*sizeof(real)), cudaMemcpyDeviceToHost);
    cudaMemcpy(HHost, H, (size_t)(n*k*sizeof(real)), cudaMemcpyDeviceToHost);
}

int ArgPos(char *str, int argc, char **argv){
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])){
            if (a == argc - 1){
                printf("Argument missing for %s\n", str);
                exit(1);
            }
        return a;
    }
    return -1;
}

int main(int argc, char **argv){
    int i, j;
    if(argc == 1){
        printf("NMF: Non-negative Matrix Factorization\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse data from <file> to train the model;\n");
        printf("\t-factor <int>\n");
        printf("\t\tfactor number; default is 3\n");
        printf("\t-maxiter <int>\n");
        printf("\t\tmaxiter for main loop; default is 10\n");
        printf("\t-timelimit <int>\n");
        printf("\t\ttimelimit for training; default is 100s\n");
        printf("\t-gpuid <int>\n");
        printf("\t\twhich gpu to use; default is 0\n");
        printf("\nExamples:\n");
        printf("./NMF_gd -train test.txt -factor 3 -maxiter 10 -timelimit 100 -gpuid 0\n\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(filename, argv[i + 1]);
    if ((i = ArgPos((char *)"-factor", argc, argv)) > 0) n = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-maxiter", argc, argv)) > 0) maxiter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-timelimit", argc, argv)) > 0) timelimit = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-gpuid", argc, argv)) > 0) gpuid = atoi(argv[i + 1]);

    initVaribles();
    shipping();
    NMF();

    //save result
    backHost();
    FILE *f = fopen("W.txt", "w");
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++)
            fprintf(f, "%.4f ", *(WHost + IDX2C(i,j,m)));
        fprintf(f, "\n");
    }
    fclose(f);
    f = fopen("H.txt", "w");
    for(i = 0; i < n; i++){
        for(j = 0; j < k; j++)
            fprintf(f, "%.4f ", *(HHost + IDX2C(i,j,n)));
        fprintf(f, "\n");
    }
    fclose(f);

    /*
    printf("Result:\n");
    printf("W:\n");
    outPutMatrix(m, n, WHost);
    printf("H:\n");
    outPutMatrix(n, k, HHost);
    */

    //slowTest
    /*
    real *Vdense, *VdenseHost=0;
    cudaMalloc((void**)&Vdense, m*k*sizeof(real));
    VdenseHost = (real *)malloc(m*k*sizeof(*VdenseHost));
    cublasSgemm(handle_blas, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &one, W, m, H, n, &zero, Vdense, m);
    cudaMemcpy(VdenseHost, Vdense, (size_t)(m*k*sizeof(real)), cudaMemcpyDeviceToHost);
    printf("WH:\n");
    outPutMatrix(m, k, VdenseHost);
    */

    CLEANUP("end.");
    return 0;
}
