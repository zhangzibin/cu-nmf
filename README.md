# cu-nmf
NMF(Non-negative Matrix Factorization) based on cuda, with sparse matrix as input.

**NMF_pgd.cu** This code solves NMF by alternative non-negative least squares using projected gradients. It's an implementation of [Projected gradient methods for non-negative matrix factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf). You may read the paper for more details.


**NMF_gd.cu** Directly GD(gradient descent) with fixed learning rate is only a simple demo for study. Using alternating non-negative least square to slove NMF problem, GD is **not a correct method** because of the projection while training. 

# Future Work
I will extend the algorithm to work on multi-GPUs in the future(not soon, however). Aim at large scale datasets.   

# Requirements
The code is base on cuda, cuBlas and cuSparse precisely. Please get cuda from Nvidia's website, https://developer.nvidia.com/cuda-downloads.


# Usage
The input matrix is store in a .txt file in sparse format. The first line is row and column number. Then Each line is a non-zero value with its row index and column index. All numbers are split by a blank. The Index is 0 based. For example, text.txt. Results will be saved in two files, W.txt and H.txt in dense format.

You should use nvcc to compile the code, so make sure cuda is installed and environment is correctly setted.

```bash
$ make
$ ./NMF_pgd -train test.txt
```
test.txt is the file storing matrix V in sparse format.

# Options for NMF_pgd
- **-factor** Factor number, which is n in fractorization m*k=(m*n)(n*k), default is 3.
- **-maxiter** Max iter number for alternating update, default is 100. 
- **-timelimit** Sometimes the algorithm takes a long time to converge, you may want to stop early, default is 1000s.
- **-gpuid** Choose the gpu device to use, default is 0.


