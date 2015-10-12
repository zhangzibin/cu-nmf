# cu-nmf
NMF(Non-negative Matrix Factorization) based on cuda, with sparse matrix as input.

**NMF_pgd.cu**  This code solves NMF by alternative non-negative least squares using projected gradients. It's a implementation of [Projected gradient methods for non-negative matrix factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf).


**NMF_gd.cu** Directly GD(gradient descent) with fixed learning rate is only a simple demo for study. Using alternating non-negative least square to slove NMF problem, GD is not a correct method because of the projection while training. 

# Future Work
If time permits, I will implement other algorithms(multiplicative update rules, and multi-GPU support) in the future(not soon, however). 

# Requirements
The code is base on cuda, cuBlas and cuSparse precisely. Please get cuda from Nvidia's website, https://developer.nvidia.com/cuda-downloads.


# Usage
The input matrix is store in a .txt file in sparse format. The first line is row and column number. Then Each line is a non-zero value with its row index and column index. All numbers are split by a blank. The Index is 0 based. For example, text.txt.

You should use nvcc to compile the code, so make sure cuda is installed and environment is correctly setted.

```bash
$ make
$ ./NMF_pgd -train test.txt
```
