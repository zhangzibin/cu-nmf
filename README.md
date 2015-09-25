# cu-nmf
NMF(Non-negative Matrix Factorization) based on cuda, with sparse matrix as input.

Only GD(gradient descent) with fixed learning rate has supported by now for its simplicity. GD is not a fully correct method for NMF because the projection while training. So the result sometimes very poor. The main algorithm is alternating non-negative least square. 


# Future Work
If time permits, I will implement other algorithms(projected gradient methods, multiplicative update rules, and multi-GPU support) in the future(not soon, however). 

# Requirements
The code is base on cuda, cuBlas and cuSparse precisely. Please get cuda from Nvidia's website, https://developer.nvidia.com/cuda-downloads.


# Usage
The input matrix is store in a .txt file in sparse format. The first line is row and column number. Then Each line is a non-zero value with its row index and column index. All numbers are split by a blank. The Index is 0 based. For example, text.txt.

You should use nvcc to compile the code, so make sure cuda is installed and environment is correctly setted.

```bash
$ make
$ ./NMF_sgd -train test.txt
```

You should notice it's hard for GD to turn parameters. -lrate and -iterMain -iterSub should be turned according your data.
