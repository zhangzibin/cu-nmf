# cu-nmf
NMF(Non-negative Matrix Factorization) based on cuda, with sparse matrix as input. Aim at large scale data and fast training.
WARNING: The code is still developing and full of bugs!


# Requirements
The code is base on cuda, cuBlas and cuSparse precisely. Please get cuda from Nvidia's website, https://developer.nvidia.com/cuda-downloads.

# Future Work
Only SGD with fixed learning rate has supported by now for its simplicity. The main algorithm is alternating non-negative least square.

If time permits, I will implement other algorithms(projected gradient methods, multiplicative update rules, and multi-GPU support) in the future.


# Usage
The input matrix is store in a .txt file in sparse format. The firsr line is row and column number. Then Each line is a non-zero value with its row index and column index. All numbers are split by a blank. The Index is 0 based. For example, text.txt.

You should use nvcc to compile the code, so make sure cuda is installed and environment is correctly setted.

```bash
$ make
$ ./NMF_sgd -train test.txt
```
