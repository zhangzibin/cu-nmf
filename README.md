# cu-nmf
NMF based on cuda, sparse matrix as input.
I implement the Non-negative Matrix Factorization (NMF) algorithm with sparse matrix as input based on cuda. Aim at large scale data and fast training.


# Requirements
The code is base on cuda, cuBlas and cuSparse precisely. Please get cuda from Nvidia, https://developer.nvidia.com/cuda-downloads.


# Future Work
Only SGD with fixed learning rate is support by now for its simplicity. If time permits, I will implement other algorithms(projected gradient methods, multiplicative update rules, and multi-GPU support) in the feature.


# Usage
```bash
$ make
$ ./NMF_sgd -train test.txt
```
